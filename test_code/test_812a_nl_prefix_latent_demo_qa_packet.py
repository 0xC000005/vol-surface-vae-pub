import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_demo_qa_packet import (
    build_gates,
    build_qa_packet,
    overall_status,
    preflight_snapshot,
    render_markdown,
    smoke_snapshot,
    staging_snapshot,
)


def _preflight_report(status: str = "pass") -> dict:
    return {
        "status": status,
        "summary": {
            "required_artifact_count": 25,
            "present_artifact_count": 25 if status == "pass" else 24,
            "missing_artifact_count": 0 if status == "pass" else 1,
            "total_required_bytes": 52835108,
            "unsafe_staged_count": 0,
            "openai_key_required": False,
        },
        "secrets": {
            "env_file_exists": True,
            "env_file_git_ignored": True,
            "openai_api_key_present": False,
        },
        "artifacts": {"missing": [] if status == "pass" else ["missing.pt"]},
    }


def _staging_manifest(private: bool = False) -> dict:
    copied_source = [{"path": "README.md"}, {"path": "experiments/app.py"}]
    if private:
        copied_source.append({"path": "paper/private.md"})
    return {
        "status": "pass",
        "stage_root": "/tmp/stage",
        "summary": {
            "copied_source_count": len(copied_source),
            "copied_artifact_count": 25,
            "total_source_bytes": 1000,
            "total_artifact_bytes": 52835108,
        },
        "preflight": {"status": "pass"},
        "copied_source": copied_source,
        "copied_artifacts": [
            {"path": "models/backfill/demo/best_model.pt"},
            {"path": "data/vol_surface_with_ret.npz"},
        ],
    }


def _smoke_report(*, mode: str, live: bool = False, fan_traces: int = 8) -> dict:
    return {
        "status": "ok",
        "auth_used": False,
        "mode": mode,
        "url": "http://127.0.0.1:7863",
        "condition_source": "external_condition_report",
        "condition_only_validation_status": "pass" if live else "",
        "selected_start_status": "pass",
        "overall_status": "pass",
        "support_alignment_status": "pass",
        "support_prior_mode": "soft_topk_combined",
        "support_candidate_count": 8,
        "market_implications": [
            {
                "market": "GOLD",
                "direction": "up",
                "horizon": "current_state",
                "target_use": "support_prior",
            }
        ],
        "condition_only_forward_warning_count": 1 if live else 0,
        "forward_warnings": [
            {
                "phrase": "forward risk",
                "handling": "ignore_for_conditioning",
                "severity": "warning",
            }
        ],
        "fan_market": "SPX",
        "fan_trace_count": fan_traces,
        "redraw_market": "IV_ATM_3M",
        "redraw_trace_count": 8,
        "condition_dim": 128,
        "embedding_dim": 1536,
        "embedding_model": "text-embedding-3-small",
        "grounding_model": "gpt-5.4-mini",
        "openai_response_id": "resp_test" if live else "",
        "openai_usage": {"total_tokens": 1943} if live else {},
        "prefix_report_path": "prefix.json",
        "condition_report_path": "condition.json",
        "support_top_candidates": [
            {
                "rank": 1,
                "window_id": "joint39_val_0092",
                "window_index": 47,
                "weight": 0.17,
                "history_start_date": "2016-04-27",
                "history_end_date": "2016-06-08",
                "forecast_start_date": "2016-06-09",
                "forecast_end_date": "2016-07-21",
            }
        ],
    }


def _auth_smoke_report() -> dict:
    report = _smoke_report(mode="cached_casebook")
    report["auth_used"] = True
    return report


def test_snapshots_extract_demo_readiness_evidence() -> None:
    preflight = preflight_snapshot(_preflight_report())
    staging = staging_snapshot(_staging_manifest())
    live = smoke_snapshot(_smoke_report(mode="live_condition_only", live=True))

    assert preflight["present_artifact_count"] == 25
    assert preflight["env_file_git_ignored"] is True
    assert staging["private_path_violation_count"] == 0
    assert staging["copied_artifact_count"] == 25
    assert live["openai_total_tokens"] == 1943
    assert live["warning_only_count"] == 1
    assert live["support_top_windows"][0]["window_id"] == "joint39_val_0092"


def test_gates_pass_for_clean_staged_cached_and_live_evidence() -> None:
    gates = build_gates(
        preflight=preflight_snapshot(_preflight_report()),
        staging=staging_snapshot(_staging_manifest()),
        cached=smoke_snapshot(_smoke_report(mode="cached_casebook")),
        live=smoke_snapshot(_smoke_report(mode="live_condition_only", live=True)),
    )

    assert overall_status(gates) == "pass"
    assert {gate["name"] for gate in gates} >= {
        "artifact_bundle_complete",
        "staged_tree_clean",
        "cached_demo_path",
        "live_openai_condition_path",
        "future_language_warning_only",
    }
    assert all(gate["status"] == "pass" for gate in gates)


def test_gates_include_authenticated_cached_demo_when_supplied() -> None:
    gates = build_gates(
        preflight=preflight_snapshot(_preflight_report()),
        staging=staging_snapshot(_staging_manifest()),
        cached=smoke_snapshot(_smoke_report(mode="cached_casebook")),
        live=smoke_snapshot(_smoke_report(mode="live_condition_only", live=True)),
        auth_smoke=smoke_snapshot(_auth_smoke_report()),
    )

    by_name = {gate["name"]: gate for gate in gates}
    assert overall_status(gates) == "pass"
    assert by_name["authenticated_cached_demo_path"]["status"] == "pass"
    assert "auth_used=True" in by_name["authenticated_cached_demo_path"]["evidence"]


def test_gates_fail_for_private_path_and_missing_visual_evidence() -> None:
    gates = build_gates(
        preflight=preflight_snapshot(_preflight_report()),
        staging=staging_snapshot(_staging_manifest(private=True)),
        cached=smoke_snapshot(_smoke_report(mode="cached_casebook", fan_traces=0)),
        live=smoke_snapshot(_smoke_report(mode="live_condition_only", live=True)),
    )

    by_name = {gate["name"]: gate for gate in gates}
    assert overall_status(gates) == "fail"
    assert by_name["staged_tree_clean"]["status"] == "fail"
    assert by_name["support_and_visual_evidence"]["status"] == "fail"
    assert "private_violations=1" in by_name["secret_and_paper_hygiene"]["evidence"]


def test_render_markdown_explains_contract_and_manual_qa() -> None:
    gates = build_gates(
        preflight=preflight_snapshot(_preflight_report()),
        staging=staging_snapshot(_staging_manifest()),
        cached=smoke_snapshot(_smoke_report(mode="cached_casebook")),
        live=smoke_snapshot(_smoke_report(mode="live_condition_only", live=True)),
    )
    packet = {
        "status": overall_status(gates),
        "preflight": preflight_snapshot(_preflight_report()),
        "staging": staging_snapshot(_staging_manifest()),
        "cached_smoke": smoke_snapshot(_smoke_report(mode="cached_casebook")),
        "live_smoke": smoke_snapshot(
            _smoke_report(mode="live_condition_only", live=True)
        ),
        "auth_smoke": smoke_snapshot(_auth_smoke_report()),
        "gates": gates,
    }

    markdown = render_markdown(packet)

    assert "Narrative Scenario Demo QA Packet" in markdown
    assert "future-looking phrases are warning-only" in markdown
    assert "Manual Visual QA Still Required" in markdown
    assert "Top Live Support Candidates" in markdown
    assert "Authenticated Staged Smoke" in markdown


def test_build_qa_packet_writes_json_and_markdown(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    staging_path = tmp_path / "staging.json"
    cached_path = tmp_path / "cached.json"
    live_path = tmp_path / "live.json"
    preflight_path.write_text(json.dumps(_preflight_report()), encoding="utf-8")
    staging_path.write_text(json.dumps(_staging_manifest()), encoding="utf-8")
    cached_path.write_text(
        json.dumps(_smoke_report(mode="cached_casebook")), encoding="utf-8"
    )
    live_path.write_text(
        json.dumps(_smoke_report(mode="live_condition_only", live=True)),
        encoding="utf-8",
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(_auth_smoke_report()), encoding="utf-8")

    packet = build_qa_packet(
        SimpleNamespace(
            preflight_report=str(preflight_path),
            staging_manifest=str(staging_path),
            cached_smoke=str(cached_path),
            live_smoke=str(live_path),
            auth_smoke=str(auth_path),
            output_dir=str(tmp_path / "out"),
        )
    )

    assert packet["status"] == "pass"
    assert packet["auth_smoke"]["auth_used"] is True
    assert (tmp_path / "out" / "demo_qa_packet.json").is_file()
    assert (tmp_path / "out" / "demo_qa_packet.md").is_file()
