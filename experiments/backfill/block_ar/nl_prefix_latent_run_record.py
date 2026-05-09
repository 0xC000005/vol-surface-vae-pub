#!/usr/bin/env python
"""Compact per-run records for prefix-latent narrative scenario runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_head(repo_root: str | Path = ".") -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _artifact_rows(paths: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in sorted(paths.items()):
        path_text = str(value or "")
        if not path_text:
            continue
        path = Path(path_text)
        row = {"key": str(key), "path": path_text, "exists": path.is_file()}
        if path.is_file():
            row["bytes"] = int(path.stat().st_size)
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def _selected_variant(report: Mapping[str, Any]) -> dict[str, Any]:
    for row in _as_list(report.get("variant_rows")):
        item = _as_dict(row)
        if item.get("is_operational") or item.get("case_role") == "operational_selected_start":
            return item
    rows = _as_list(report.get("variant_rows"))
    return _as_dict(rows[0]) if rows else {}


def _support_candidates(report: Mapping[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    prior = _as_dict(_as_dict(report.get("cached_query")).get("memory_prior"))
    candidates = []
    for row in _as_list(prior.get("candidate_details"))[:limit]:
        item = _as_dict(row)
        candidates.append(
            {
                "rank": item.get("rank"),
                "window_id": item.get("window_id"),
                "bridge_local_index": item.get("bridge_local_index"),
                "source_index": item.get("source_index"),
                "history_start_date": item.get("history_start_date"),
                "history_end_date": item.get("history_end_date"),
                "manifest_split": item.get("manifest_split"),
                "weight": item.get("weight"),
                "memory_support_cosine": item.get("memory_support_cosine"),
                "start_distance_z": item.get("start_distance_z"),
                "recent_prefix_alignment_score": item.get(
                    "recent_prefix_alignment_score"
                ),
                "recent_prefix_mismatches": item.get("recent_prefix_mismatches"),
                "combined_score": item.get("combined_score"),
            }
        )
    return candidates


def _forward_warning_count(report: Mapping[str, Any]) -> int:
    grounding = _as_dict(_as_dict(report.get("cached_query")).get("grounding"))
    warnings = _as_list(grounding.get("non_conditioning_forward_language"))
    return len(warnings)


def _condition_implication_count(report: Mapping[str, Any]) -> int:
    grounding = _as_dict(_as_dict(report.get("cached_query")).get("grounding"))
    return len(_as_list(grounding.get("market_implications")))


def _record_id(report: Mapping[str, Any], source_commit: str) -> str:
    artifacts = _as_dict(report.get("artifact_paths"))
    seed = "|".join(
        [
            source_commit,
            str(artifacts.get("report", "")),
            str(_as_dict(report.get("cached_query")).get("condition_source", "")),
            str(_selected_variant(report).get("variant", "")),
        ]
    )
    return sha256_text(seed)[:16]


def build_prefix_run_record(
    report: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    artifacts = _as_dict(report.get("artifact_paths"))
    query = _as_dict(report.get("cached_query"))
    gate = _as_dict(report.get("validation_gate"))
    generation = _as_dict(report.get("generation"))
    selected = _selected_variant(report)
    story_text = str(query.get("narrative_text", ""))
    source_commit = git_head(repo_root)
    record = {
        "record_id": _record_id(report, source_commit),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "record_type": "prefix_latent_narrative_scenario_run",
        "status": str(
            gate.get(
                "selected_start_status",
                gate.get("operational_status", gate.get("overall_status", "unknown")),
            )
        ),
        "condition": {
            "source": str(query.get("condition_source", "")),
            "window_id": str(query.get("window_id", "")),
            "kind": str(query.get("kind", "")),
            "text_memory_dim": _as_int(query.get("text_memory_dim")),
            "narrative_text_sha256": sha256_text(story_text) if story_text else "",
            "narrative_text_characters": len(story_text),
            "market_implication_count": _condition_implication_count(report),
            "forward_warning_count": _forward_warning_count(report),
        },
        "selected_start": {
            "variant": str(selected.get("variant", "")),
            "window_id": str(selected.get("start_window_id", "")),
            "window_index": selected.get("start_window_index"),
            "manifest_split": str(selected.get("start_manifest_split", "")),
            "selection_method": str(selected.get("start_selection_method", "")),
            "start_distance_z": selected.get("start_distance_z"),
            "memory_support_cosine": selected.get("memory_support_cosine"),
        },
        "validation": {
            "overall_status": str(gate.get("overall_status", "")),
            "selected_start_status": str(gate.get("selected_start_status", "")),
            "operational_status": str(gate.get("operational_status", "")),
            "stress_status": str(gate.get("stress_status", "")),
            "endpoint_max_abs_error": gate.get("endpoint_max_abs_error"),
        },
        "support": {
            "candidate_count": len(_support_candidates(report)),
            "top_candidates": _support_candidates(report),
        },
        "generation": {
            "generated_state_shape": generation.get("generated_state_shape"),
            "finite_rate": generation.get("finite_rate"),
            "sample_count": generation.get("sample_count"),
            "rollout_temperature": generation.get("rollout_temperature"),
        },
        "artifacts": _artifact_rows(artifacts),
    }
    product_gate = _as_dict(report.get("condition_only_product_gate"))
    decision = _as_dict(product_gate.get("production_decision"))
    if decision:
        record["product_decision"] = {
            "decision": str(decision.get("decision", "")),
            "ui_guidance": str(decision.get("ui_guidance", decision.get("reason", ""))),
        }
    return record


def write_prefix_run_record(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    record = build_prefix_run_record(report, repo_root=repo_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "prefix_latent_run_record.json"
    markdown_path = output / "prefix_latent_run_record.md"
    record["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(record).rstrip() + "\n", encoding="utf-8")
    return record


def render_markdown(record: Mapping[str, Any]) -> str:
    condition = _as_dict(record.get("condition"))
    selected = _as_dict(record.get("selected_start"))
    validation = _as_dict(record.get("validation"))
    support = _as_dict(record.get("support"))
    lines = [
        "# Prefix-Latent Run Record",
        "",
        f"Record: `{record.get('record_id', '')}`",
        f"Status: `{record.get('status', '')}`",
        f"Source commit: `{record.get('source_commit', '')}`",
        "",
        "## Condition",
        "",
        f"- Source: `{condition.get('source', '')}`",
        f"- Text memory dim: `{condition.get('text_memory_dim', 0)}`",
        f"- Market implications: `{condition.get('market_implication_count', 0)}`",
        f"- Forward warnings: `{condition.get('forward_warning_count', 0)}`",
        f"- Narrative hash: `{condition.get('narrative_text_sha256', '')[:12]}`",
        "",
        "## Selected Start",
        "",
        f"- Variant: `{selected.get('variant', '')}`",
        f"- Window: `{selected.get('window_id', '')}`",
        f"- Method: `{selected.get('selection_method', '')}`",
        f"- Start distance z: `{selected.get('start_distance_z', '')}`",
        "",
        "## Validation",
        "",
        f"- Overall: `{validation.get('overall_status', '')}`",
        f"- Selected start: `{validation.get('selected_start_status', '')}`",
        f"- Stress: `{validation.get('stress_status', '')}`",
        "",
        "## Support",
        "",
        f"- Candidate count: `{support.get('candidate_count', 0)}`",
        "",
        "| Rank | Window | Weight | Start z | Memory cosine |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in _as_list(support.get("top_candidates")):
        item = _as_dict(row)
        lines.append(
            f"| {item.get('rank', '')} | `{item.get('window_id', '')}` | "
            f"{item.get('weight', '')} | {item.get('start_distance_z', '')} | "
            f"{item.get('memory_support_cosine', '')} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Hashes",
            "",
            "| Key | Exists | Bytes | SHA-256 | Path |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for row in _as_list(record.get("artifacts")):
        item = _as_dict(row)
        lines.append(
            f"| `{item.get('key', '')}` | `{item.get('exists', False)}` | "
            f"{item.get('bytes', 0)} | `{str(item.get('sha256', ''))[:12]}` | "
            f"`{item.get('path', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Sensitive Data Contract",
            "",
            (
                "Raw narrative text is not stored in this record. The record "
                "keeps only a SHA-256 hash and character count for narrative "
                "traceability."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Prefix-latent report JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    record = write_prefix_run_record(
        _load_json(args.report),
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "record_id": record["record_id"],
                "status": record["status"],
                "json": record["artifact_paths"]["json"],
                "markdown": record["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
