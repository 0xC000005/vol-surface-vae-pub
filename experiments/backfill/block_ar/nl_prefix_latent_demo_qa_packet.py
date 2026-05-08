#!/usr/bin/env python
"""Build a QA packet for the narrative prefix-latent Gradio demo."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PREFLIGHT_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_demo_preflight_834a_complete_bundle/demo_preflight_report.json"
)
DEFAULT_STAGING_MANIFEST = "/tmp/nl_prefix_latent_demo_stage_832a/demo_staging_manifest.json"
DEFAULT_CACHED_SMOKE = (
    "/tmp/nl_prefix_latent_demo_stage_832a/experiments/backfill/block_ar/"
    "nl_scenario_demo_outputs/prefix_latent_staged_gradio_api_smoke_833f_complete_bundle/"
    "gradio_api_smoke_summary.json"
)
DEFAULT_LIVE_SMOKE = (
    "/tmp/nl_prefix_latent_demo_stage_832a/experiments/backfill/block_ar/"
    "nl_scenario_demo_outputs/prefix_latent_staged_gradio_api_live_testflight_835a_secret_env/"
    "gradio_api_smoke_summary.json"
)
DEFAULT_AUTH_SMOKE = ""
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_demo_qa_packet_836a"
)

PRIVATE_EXACT_PATHS = {".env"}
PRIVATE_PREFIXES = (
    ".agents/",
    ".claude/",
    ".venv/",
    "autoresearch-session/",
    "paper/",
)
PRIVATE_SUFFIXES = (".pdf",)


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


def _path_value(row: Any) -> str:
    return str(_as_dict(row).get("path", "")).replace("\\", "/")


def _is_private_path(path: str) -> bool:
    if path in PRIVATE_EXACT_PATHS:
        return True
    if any(path.startswith(prefix) for prefix in PRIVATE_PREFIXES):
        return True
    return any(path.endswith(suffix) for suffix in PRIVATE_SUFFIXES)


def preflight_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    secrets = _as_dict(report.get("secrets"))
    artifacts = _as_dict(report.get("artifacts"))
    return {
        "status": str(report.get("status", "")),
        "required_artifact_count": _as_int(summary.get("required_artifact_count")),
        "present_artifact_count": _as_int(summary.get("present_artifact_count")),
        "missing_artifact_count": _as_int(summary.get("missing_artifact_count")),
        "total_required_bytes": _as_int(summary.get("total_required_bytes")),
        "unsafe_staged_count": _as_int(summary.get("unsafe_staged_count")),
        "env_file_exists": bool(secrets.get("env_file_exists")),
        "env_file_git_ignored": secrets.get("env_file_git_ignored"),
        "openai_key_required": bool(summary.get("openai_key_required")),
        "openai_key_present": bool(secrets.get("openai_api_key_present")),
        "missing_artifacts": [str(path) for path in _as_list(artifacts.get("missing"))],
    }


def staging_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    copied_source = _as_list(manifest.get("copied_source"))
    copied_artifacts = _as_list(manifest.get("copied_artifacts"))
    copied_paths = [_path_value(row) for row in copied_source + copied_artifacts]
    private_violations = sorted(path for path in copied_paths if _is_private_path(path))
    summary = _as_dict(manifest.get("summary"))
    preflight = _as_dict(manifest.get("preflight"))
    return {
        "status": str(manifest.get("status", "")),
        "stage_root": str(manifest.get("stage_root", "")),
        "copied_source_count": _as_int(summary.get("copied_source_count")),
        "copied_artifact_count": _as_int(summary.get("copied_artifact_count")),
        "total_source_bytes": _as_int(summary.get("total_source_bytes")),
        "total_artifact_bytes": _as_int(summary.get("total_artifact_bytes")),
        "preflight_status": str(preflight.get("status", "")),
        "private_path_violation_count": len(private_violations),
        "private_path_violations": private_violations[:20],
    }


def smoke_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    warnings = _as_list(report.get("forward_warnings"))
    warning_only_count = sum(
        1
        for row in warnings
        if str(_as_dict(row).get("handling", "")) == "ignore_for_conditioning"
    )
    usage = _as_dict(report.get("openai_usage"))
    return {
        "status": str(report.get("status", "")),
        "auth_used": bool(report.get("auth_used", False)),
        "mode": str(report.get("mode", "")),
        "url": str(report.get("url", "")),
        "condition_source": str(report.get("condition_source", "")),
        "condition_only_validation_status": str(
            report.get("condition_only_validation_status", "")
        ),
        "selected_start_status": str(report.get("selected_start_status", "")),
        "overall_status": str(report.get("overall_status", "")),
        "support_alignment_status": str(report.get("support_alignment_status", "")),
        "support_prior_mode": str(report.get("support_prior_mode", "")),
        "support_candidate_count": _as_int(report.get("support_candidate_count")),
        "market_implication_count": len(_as_list(report.get("market_implications"))),
        "forward_warning_count": max(
            _as_int(report.get("condition_only_forward_warning_count")),
            len(warnings),
        ),
        "warning_only_count": warning_only_count,
        "fan_market": str(report.get("fan_market", "")),
        "fan_trace_count": _as_int(report.get("fan_trace_count")),
        "redraw_market": str(report.get("redraw_market", "")),
        "redraw_trace_count": _as_int(report.get("redraw_trace_count")),
        "condition_dim": _as_int(report.get("condition_dim")),
        "embedding_dim": _as_int(report.get("embedding_dim")),
        "embedding_model": str(report.get("embedding_model", "")),
        "grounding_model": str(report.get("grounding_model", "")),
        "openai_response_id_present": bool(str(report.get("openai_response_id", ""))),
        "openai_total_tokens": _as_int(usage.get("total_tokens")),
        "prefix_report_path": str(report.get("prefix_report_path", "")),
        "condition_report_path": str(report.get("condition_report_path", "")),
        "support_top_windows": [
            {
                "rank": _as_dict(row).get("rank"),
                "window_id": _as_dict(row).get("window_id"),
                "window_index": _as_dict(row).get("window_index"),
                "weight": _as_dict(row).get("weight"),
                "history_start_date": _as_dict(row).get("history_start_date"),
                "history_end_date": _as_dict(row).get("history_end_date"),
                "forecast_start_date": _as_dict(row).get("forecast_start_date"),
                "forecast_end_date": _as_dict(row).get("forecast_end_date"),
            }
            for row in _as_list(report.get("support_top_candidates"))[:8]
        ],
    }


def _gate(
    name: str,
    passed: bool,
    evidence: str,
    *,
    blocking: bool = True,
    warning: bool = False,
) -> dict[str, Any]:
    status = "pass" if passed else ("warn" if warning else "fail")
    return {
        "name": name,
        "status": status,
        "blocking": bool(blocking),
        "evidence": evidence,
    }


def build_gates(
    *,
    preflight: Mapping[str, Any],
    staging: Mapping[str, Any],
    cached: Mapping[str, Any],
    live: Mapping[str, Any],
    auth_smoke: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    gates = [
        _gate(
            "artifact_bundle_complete",
            preflight["status"] == "pass"
            and preflight["missing_artifact_count"] == 0
            and preflight["present_artifact_count"] == preflight["required_artifact_count"],
            (
                f"{preflight['present_artifact_count']}/"
                f"{preflight['required_artifact_count']} artifacts, "
                f"{preflight['total_required_bytes']} bytes"
            ),
        ),
        _gate(
            "staged_tree_clean",
            staging["status"] == "pass"
            and staging["preflight_status"] == "pass"
            and staging["private_path_violation_count"] == 0,
            (
                f"stage={staging['stage_root']}, "
                f"source={staging['copied_source_count']}, "
                f"artifacts={staging['copied_artifact_count']}, "
                f"private_violations={staging['private_path_violation_count']}"
            ),
        ),
        _gate(
            "cached_demo_path",
            cached["status"] == "ok"
            and cached["selected_start_status"] == "pass"
            and cached["overall_status"] == "pass"
            and cached["condition_source"] == "external_condition_report"
            and cached["support_candidate_count"] > 0,
            (
                f"mode={cached['mode']}, selected_start={cached['selected_start_status']}, "
                f"overall={cached['overall_status']}, support={cached['support_candidate_count']}"
            ),
        ),
        _gate(
            "live_openai_condition_path",
            live["status"] == "ok"
            and live["mode"] == "live_condition_only"
            and live["condition_only_validation_status"] == "pass"
            and live["selected_start_status"] == "pass"
            and live["overall_status"] == "pass"
            and live["openai_response_id_present"],
            (
                f"condition={live['condition_only_validation_status']}, "
                f"selected_start={live['selected_start_status']}, "
                f"tokens={live['openai_total_tokens']}"
            ),
        ),
        _gate(
            "support_and_visual_evidence",
            cached["fan_trace_count"] > 0
            and cached["redraw_trace_count"] > 0
            and live["fan_trace_count"] > 0
            and live["redraw_trace_count"] > 0
            and live["market_implication_count"] > 0,
            (
                f"cached fan/redraw={cached['fan_trace_count']}/{cached['redraw_trace_count']}, "
                f"live fan/redraw={live['fan_trace_count']}/{live['redraw_trace_count']}, "
                f"live implications={live['market_implication_count']}"
            ),
        ),
        _gate(
            "future_language_warning_only",
            live["forward_warning_count"] > 0
            and live["warning_only_count"] == live["forward_warning_count"],
            (
                f"forward_warnings={live['forward_warning_count']}, "
                f"warning_only={live['warning_only_count']}"
            ),
        ),
        _gate(
            "secret_and_paper_hygiene",
            preflight["env_file_git_ignored"] is not False
            and staging["private_path_violation_count"] == 0,
            (
                f"env_ignored={preflight['env_file_git_ignored']}, "
                f"private_violations={staging['private_path_violation_count']}"
            ),
        ),
    ]
    if auth_smoke is not None:
        gates.append(
            _gate(
                "authenticated_cached_demo_path",
                auth_smoke["status"] == "ok"
                and auth_smoke["auth_used"]
                and auth_smoke["selected_start_status"] == "pass"
                and auth_smoke["overall_status"] == "pass"
                and auth_smoke["support_candidate_count"] > 0
                and auth_smoke["fan_trace_count"] > 0
                and auth_smoke["redraw_trace_count"] > 0,
                (
                    f"auth_used={auth_smoke['auth_used']}, "
                    f"selected_start={auth_smoke['selected_start_status']}, "
                    f"support={auth_smoke['support_candidate_count']}, "
                    f"fan/redraw={auth_smoke['fan_trace_count']}/"
                    f"{auth_smoke['redraw_trace_count']}"
                ),
            )
        )
    return gates


def overall_status(gates: Sequence[Mapping[str, Any]]) -> str:
    blocking_failures = [
        gate
        for gate in gates
        if gate.get("blocking") and str(gate.get("status", "")) == "fail"
    ]
    if blocking_failures:
        return "fail"
    if any(str(gate.get("status", "")) == "warn" for gate in gates):
        return "warn"
    return "pass"


def render_markdown(packet: Mapping[str, Any]) -> str:
    preflight = _as_dict(packet.get("preflight"))
    staging = _as_dict(packet.get("staging"))
    cached = _as_dict(packet.get("cached_smoke"))
    live = _as_dict(packet.get("live_smoke"))
    auth_smoke = _as_dict(packet.get("auth_smoke"))
    gates = _as_list(packet.get("gates"))
    lines = [
        "# Narrative Scenario Demo QA Packet",
        "",
        f"Status: `{packet.get('status', '')}`",
        "",
        "## Product Contract Under Test",
        "",
        (
            "A risk manager enters a current/recent market story, fixes or "
            "accepts a joint39 starting level, and receives a frozen-generator "
            "30-day scenario distribution. The narrative is used to form "
            "condition-only market implications and a start-compatible soft "
            "analogue mixture; future-looking phrases are warning-only."
        ),
        "",
        "## Gates",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for gate in gates:
        gate_map = _as_dict(gate)
        lines.append(
            f"| {gate_map.get('name', '')} | `{gate_map.get('status', '')}` | "
            f"{gate_map.get('evidence', '')} |"
        )
    lines.extend(
        [
            "",
            "## Bundle And Staging",
            "",
            f"- Required artifacts: `{preflight.get('present_artifact_count', 0)}/"
            f"{preflight.get('required_artifact_count', 0)}`",
            f"- Required bytes: `{preflight.get('total_required_bytes', 0)}`",
            f"- Staged root: `{staging.get('stage_root', '')}`",
            f"- Staged source files: `{staging.get('copied_source_count', 0)}`",
            f"- Staged artifact files: `{staging.get('copied_artifact_count', 0)}`",
            f"- Private path violations: `{staging.get('private_path_violation_count', 0)}`",
            "",
            "## Cached Demo Smoke",
            "",
            f"- Status: `{cached.get('status', '')}`",
            f"- Selected start: `{cached.get('selected_start_status', '')}`",
            f"- Overall gate: `{cached.get('overall_status', '')}`",
            f"- Support prior: `{cached.get('support_prior_mode', '')}`",
            f"- Support candidates: `{cached.get('support_candidate_count', 0)}`",
            f"- Fan redraw: `{cached.get('fan_market', '')}` "
            f"{cached.get('fan_trace_count', 0)} traces; "
            f"`{cached.get('redraw_market', '')}` "
            f"{cached.get('redraw_trace_count', 0)} traces",
            "",
            "## Live OpenAI TestFlight",
            "",
            f"- Status: `{live.get('status', '')}`",
            f"- Grounding model: `{live.get('grounding_model', '')}`",
            f"- Embedding model: `{live.get('embedding_model', '')}`",
            f"- OpenAI tokens: `{live.get('openai_total_tokens', 0)}`",
            f"- Condition validation: `{live.get('condition_only_validation_status', '')}`",
            f"- Forward warnings treated as non-conditioning: "
            f"`{live.get('warning_only_count', 0)}/"
            f"{live.get('forward_warning_count', 0)}`",
            f"- Prefix report: `{live.get('prefix_report_path', '')}`",
            f"- Condition report: `{live.get('condition_report_path', '')}`",
            "",
        ]
    )
    if auth_smoke:
        lines.extend(
            [
                "## Authenticated Staged Smoke",
                "",
                f"- Status: `{auth_smoke.get('status', '')}`",
                f"- Auth used: `{auth_smoke.get('auth_used', False)}`",
                f"- Selected start: `{auth_smoke.get('selected_start_status', '')}`",
                f"- Support candidates: `{auth_smoke.get('support_candidate_count', 0)}`",
                f"- Fan redraw: `{auth_smoke.get('fan_market', '')}` "
                f"{auth_smoke.get('fan_trace_count', 0)} traces; "
                f"`{auth_smoke.get('redraw_market', '')}` "
                f"{auth_smoke.get('redraw_trace_count', 0)} traces",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Live Support Candidates",
            "",
            "| Rank | Window | Weight | History | Forecast |",
            "| ---: | --- | ---: | --- | --- |",
        ]
    )
    for row in _as_list(live.get("support_top_windows")):
        item = _as_dict(row)
        lines.append(
            "| "
            f"{item.get('rank', '')} | "
            f"{item.get('window_id', '')} / {item.get('window_index', '')} | "
            f"{item.get('weight', '')} | "
            f"{item.get('history_start_date', '')} to {item.get('history_end_date', '')} | "
            f"{item.get('forecast_start_date', '')} to {item.get('forecast_end_date', '')} |"
        )
    lines.extend(
        [
            "",
            "## Manual Visual QA Still Required",
            "",
            "- Capture the readiness panel before and after a cached run.",
            "- Capture live condition-only implications and warning tables.",
            "- Confirm the analogue candidate table shows multiple weighted candidates.",
            "- Switch the factor fan chart and IV-cell fan chart after generation.",
            "- Confirm individual analogue path traces and selected realized overlays render.",
            "- Archive the generated JSON and Markdown report paths for the demo run.",
            "",
            "## Production Gaps",
            "",
            "- Private hosting needs authentication and platform-secret configuration.",
            "- Live run artifacts need persistent audit storage outside local `/tmp`.",
            "- Browser screenshot QA should be automated once a browser test dependency is accepted.",
            "- Narrative coverage still needs a larger hard-case casebook before external use.",
        ]
    )
    return "\n".join(lines)


def build_qa_packet(args: argparse.Namespace) -> dict[str, Any]:
    preflight_report = _load_json(args.preflight_report)
    staging_manifest = _load_json(args.staging_manifest)
    cached_report = _load_json(args.cached_smoke)
    live_report = _load_json(args.live_smoke)
    auth_report = None
    auth_smoke_path = str(getattr(args, "auth_smoke", "") or "").strip()
    if auth_smoke_path:
        auth_report = _load_json(auth_smoke_path)

    preflight = preflight_snapshot(preflight_report)
    staging = staging_snapshot(staging_manifest)
    cached = smoke_snapshot(cached_report)
    live = smoke_snapshot(live_report)
    auth_smoke = smoke_snapshot(auth_report) if auth_report is not None else None
    gates = build_gates(
        preflight=preflight,
        staging=staging,
        cached=cached,
        live=live,
        auth_smoke=auth_smoke,
    )
    output_dir = Path(args.output_dir)
    artifact_paths = {
        "summary_json": str(output_dir / "demo_qa_packet.json"),
        "summary_markdown": str(output_dir / "demo_qa_packet.md"),
    }
    packet = {
        "status": overall_status(gates),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "QA packet built from existing staged preflight, cached smoke, and "
            "live OpenAI smoke artifacts. It does not launch Gradio or call OpenAI."
        ),
        "inputs": {
            "preflight_report": str(args.preflight_report),
            "staging_manifest": str(args.staging_manifest),
            "cached_smoke": str(args.cached_smoke),
            "live_smoke": str(args.live_smoke),
            "auth_smoke": auth_smoke_path,
        },
        "preflight": preflight,
        "staging": staging,
        "cached_smoke": cached,
        "live_smoke": live,
        "gates": gates,
        "artifact_paths": artifact_paths,
    }
    if auth_smoke is not None:
        packet["auth_smoke"] = auth_smoke
    _write_json(artifact_paths["summary_json"], packet)
    Path(artifact_paths["summary_markdown"]).write_text(
        render_markdown(packet).rstrip() + "\n",
        encoding="utf-8",
    )
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", default=DEFAULT_PREFLIGHT_REPORT)
    parser.add_argument("--staging-manifest", default=DEFAULT_STAGING_MANIFEST)
    parser.add_argument("--cached-smoke", default=DEFAULT_CACHED_SMOKE)
    parser.add_argument("--live-smoke", default=DEFAULT_LIVE_SMOKE)
    parser.add_argument("--auth-smoke", default=DEFAULT_AUTH_SMOKE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    packet = build_qa_packet(args)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "summary_json": packet["artifact_paths"]["summary_json"],
                "summary_markdown": packet["artifact_paths"]["summary_markdown"],
                "gate_count": len(packet["gates"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
