#!/usr/bin/env python
"""Build a safe run registry for narrative demo evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_demo_run_registry_840a"
)
DEFAULT_QA_PACKET = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_demo_qa_packet_837b_auth_staged/demo_qa_packet.json"
)
DEFAULT_AUDIT_MANIFEST = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_demo_audit_manifest_838a_auth_packet/demo_audit_manifest.json"
)
DEFAULT_BROWSER_QA = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_browser_qa_839g_mobile_clean/browser_qa_report.json"
)
DEFAULT_PREFIX_RUN_RECORD = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo/prefix_latent_live_smoke/cached_casebook_run/"
    "prefix_latent_condition_only_report_823b_safe_haven/run_record/"
    "prefix_latent_run_record.json"
)
DEFAULT_EVIDENCE = (
    f"qa_packet:qa_packet:{DEFAULT_QA_PACKET}",
    f"audit_manifest:audit_manifest:{DEFAULT_AUDIT_MANIFEST}",
    f"browser_qa:browser_qa:{DEFAULT_BROWSER_QA}",
    f"prefix_run_record:prefix_run_record:{DEFAULT_PREFIX_RUN_RECORD}",
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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_root: str | Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def parse_evidence_spec(value: str) -> dict[str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "evidence specs must be formatted as name:type:path, "
            f"got {value!r}"
        )
    name, kind, path = parts
    return {"name": name, "type": kind, "path": path}


def qa_packet_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = [_as_dict(row) for row in _as_list(payload.get("gates"))]
    failed = [str(row.get("name", "")) for row in gates if row.get("status") == "fail"]
    warned = [str(row.get("name", "")) for row in gates if row.get("status") == "warn"]
    live = _as_dict(payload.get("live_smoke"))
    auth = _as_dict(payload.get("auth_smoke"))
    return {
        "status": str(payload.get("status", "")),
        "gate_count": len(gates),
        "pass_gate_count": sum(1 for row in gates if row.get("status") == "pass"),
        "failed_gates": failed,
        "warning_gates": warned,
        "live_openai_tokens": _as_int(live.get("openai_total_tokens")),
        "cached_support_candidates": _as_int(
            _as_dict(payload.get("cached_smoke")).get("support_candidate_count")
        ),
        "live_support_candidates": _as_int(live.get("support_candidate_count")),
        "auth_used": bool(auth.get("auth_used")),
    }


def audit_manifest_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(payload.get("summary"))
    return {
        "status": str(payload.get("status", "")),
        "input_summary_count": _as_int(summary.get("input_summary_count")),
        "found_artifact_count": _as_int(summary.get("found_artifact_count")),
        "missing_reference_count": _as_int(summary.get("missing_reference_count")),
        "total_artifact_bytes": _as_int(summary.get("total_artifact_bytes")),
        "sensitive_text_count": _as_int(summary.get("sensitive_text_count")),
        "raw_sensitive_text_retained": any(
            "text" in _as_dict(row) for row in _as_list(payload.get("sensitive_text"))
        ),
    }


def browser_qa_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    screenshots = [_as_dict(row) for row in _as_list(payload.get("screenshots"))]
    return {
        "status": str(payload.get("status", "")),
        "screenshot_count": len(screenshots),
        "pass_screenshot_count": sum(
            1 for row in screenshots if row.get("status") == "pass"
        ),
        "viewports": [
            {
                "label": str(row.get("label", "")),
                "width": _as_int(row.get("width")),
                "height": _as_int(row.get("height")),
                "bytes": _as_int(row.get("bytes")),
            }
            for row in screenshots
        ],
    }


def prefix_run_record_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    condition = _as_dict(payload.get("condition"))
    support = _as_dict(payload.get("support"))
    validation = _as_dict(payload.get("validation"))
    artifact_rows = [_as_dict(row) for row in _as_list(payload.get("artifacts"))]
    return {
        "status": str(payload.get("status", "")),
        "record_id": str(payload.get("record_id", "")),
        "record_type": str(payload.get("record_type", "")),
        "condition_source": str(condition.get("source", "")),
        "text_memory_dim": _as_int(condition.get("text_memory_dim")),
        "market_implication_count": _as_int(
            condition.get("market_implication_count")
        ),
        "forward_warning_count": _as_int(condition.get("forward_warning_count")),
        "narrative_hash_present": bool(condition.get("narrative_text_sha256")),
        "support_candidate_count": _as_int(support.get("candidate_count")),
        "overall_status": str(validation.get("overall_status", "")),
        "selected_start_status": str(validation.get("selected_start_status", "")),
        "artifact_count": len(artifact_rows),
        "hashed_artifact_count": sum(1 for row in artifact_rows if row.get("sha256")),
    }


def generic_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": str(payload.get("status", ""))}


def summarize_evidence(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "qa_packet":
        return qa_packet_summary(payload)
    if kind == "audit_manifest":
        return audit_manifest_summary(payload)
    if kind == "browser_qa":
        return browser_qa_summary(payload)
    if kind == "prefix_run_record":
        return prefix_run_record_summary(payload)
    return generic_summary(payload)


def _gate(name: str, passed: bool, evidence: str, *, warning: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else ("warn" if warning else "fail"),
        "evidence": evidence,
    }


def build_registry_gates(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_type = {str(row.get("type")): row for row in entries}
    qa = _as_dict(_as_dict(by_type.get("qa_packet")).get("summary"))
    audit = _as_dict(_as_dict(by_type.get("audit_manifest")).get("summary"))
    browser = _as_dict(_as_dict(by_type.get("browser_qa")).get("summary"))
    run_record = _as_dict(_as_dict(by_type.get("prefix_run_record")).get("summary"))
    return [
        _gate(
            "qa_packet_pass",
            qa.get("status") == "pass" and not qa.get("failed_gates"),
            (
                f"status={qa.get('status', '')}, "
                f"gates={qa.get('pass_gate_count', 0)}/{qa.get('gate_count', 0)}"
            ),
        ),
        _gate(
            "audit_manifest_pass",
            audit.get("status") == "pass"
            and audit.get("missing_reference_count") == 0
            and not audit.get("raw_sensitive_text_retained"),
            (
                f"status={audit.get('status', '')}, "
                f"artifacts={audit.get('found_artifact_count', 0)}, "
                f"missing={audit.get('missing_reference_count', 0)}, "
                f"raw_sensitive_text={audit.get('raw_sensitive_text_retained', False)}"
            ),
        ),
        _gate(
            "browser_render_pass",
            browser.get("status") == "pass"
            and browser.get("pass_screenshot_count") == browser.get("screenshot_count")
            and browser.get("screenshot_count", 0) >= 2,
            (
                f"status={browser.get('status', '')}, "
                f"screenshots={browser.get('pass_screenshot_count', 0)}/"
                f"{browser.get('screenshot_count', 0)}"
            ),
        ),
        _gate(
            "prefix_run_record_pass",
            run_record.get("status") == "pass"
            and run_record.get("record_type") == "prefix_latent_narrative_scenario_run"
            and run_record.get("selected_start_status") == "pass"
            and run_record.get("support_candidate_count", 0) > 0
            and run_record.get("narrative_hash_present"),
            (
                f"status={run_record.get('status', '')}, "
                f"selected_start={run_record.get('selected_start_status', '')}, "
                f"support={run_record.get('support_candidate_count', 0)}, "
                f"narrative_hash={run_record.get('narrative_hash_present', False)}"
            ),
        ),
    ]


def overall_status(gates: Sequence[Mapping[str, Any]]) -> str:
    if any(row.get("status") == "fail" for row in gates):
        return "fail"
    if any(row.get("status") == "warn" for row in gates):
        return "warn"
    return "pass"


def evidence_entry(spec: Mapping[str, str]) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.is_file():
        return {
            "name": spec["name"],
            "type": spec["type"],
            "path": spec["path"],
            "exists": False,
            "summary": {"status": "missing"},
        }
    payload = _load_json(path)
    return {
        "name": spec["name"],
        "type": spec["type"],
        "path": spec["path"],
        "exists": True,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "summary": summarize_evidence(spec["type"], payload),
    }


def render_markdown(registry: Mapping[str, Any]) -> str:
    lines = [
        "# Narrative Demo Run Registry",
        "",
        f"Status: `{registry['status']}`",
        "",
        "## Summary",
        "",
        f"- Source commit: `{registry.get('source_commit', '')}`",
        f"- Evidence entries: `{registry['summary']['evidence_count']}`",
        f"- Existing entries: `{registry['summary']['existing_count']}`",
        f"- Missing entries: `{registry['summary']['missing_count']}`",
        "",
        "## Gates",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for gate in registry["gates"]:
        lines.append(
            f"| `{gate['name']}` | `{gate['status']}` | {gate['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "| Name | Type | Status | Bytes | SHA-256 | Path |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in registry["evidence"]:
        summary = _as_dict(row.get("summary"))
        lines.append(
            f"| `{row['name']}` | `{row['type']}` | "
            f"`{summary.get('status', '')}` | {row.get('bytes', 0)} | "
            f"`{str(row.get('sha256', ''))[:12]}` | `{row['path']}` |"
        )
    lines.extend(
        [
            "",
            "## Sensitive Data Contract",
            "",
            (
                "The registry stores evidence summaries, file sizes, and hashes. "
                "It does not copy raw report payloads or raw risk-manager "
                "narratives. Use the audit manifest for approved artifact "
                "archival and narrative-text hashing."
            ),
        ]
    )
    return "\n".join(lines)


def build_registry(args: argparse.Namespace) -> dict[str, Any]:
    evidence_values = args.evidence if args.evidence is not None else DEFAULT_EVIDENCE
    specs = [parse_evidence_spec(value) for value in evidence_values]
    entries = [evidence_entry(spec) for spec in specs]
    missing_count = sum(1 for row in entries if not row.get("exists"))
    gates = build_registry_gates(entries)
    if missing_count:
        gates.append(
            _gate(
                "required_evidence_present",
                False,
                f"missing={missing_count}",
            )
        )
    registry = {
        "status": overall_status(gates),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_head(Path(args.repo_root)),
        "evidence": entries,
        "gates": gates,
        "summary": {
            "evidence_count": len(entries),
            "existing_count": len(entries) - missing_count,
            "missing_count": missing_count,
        },
        "artifact_paths": {
            "json": str(Path(args.output_dir) / "demo_run_registry.json"),
            "markdown": str(Path(args.output_dir) / "demo_run_registry.md"),
        },
    }
    _write_json(registry["artifact_paths"]["json"], registry)
    Path(registry["artifact_paths"]["markdown"]).write_text(
        render_markdown(registry).rstrip() + "\n",
        encoding="utf-8",
    )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evidence",
        action="append",
        default=None,
        help="Evidence spec formatted as name:type:path",
    )
    args = parser.parse_args()
    if args.evidence is None:
        args.evidence = list(DEFAULT_EVIDENCE)
    registry = build_registry(args)
    print(
        json.dumps(
            {
                "status": registry["status"],
                "evidence_count": registry["summary"]["evidence_count"],
                "json": registry["artifact_paths"]["json"],
                "markdown": registry["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
