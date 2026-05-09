import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_demo_run_registry import (
    build_registry,
    build_registry_gates,
    browser_qa_summary,
    overall_status,
    parse_evidence_spec,
    prefix_run_record_summary,
    render_markdown,
)


def _qa_packet(status: str = "pass") -> dict:
    return {
        "status": status,
        "gates": [
            {"name": "cached_demo_path", "status": "pass"},
            {"name": "live_openai_condition_path", "status": status},
        ],
        "cached_smoke": {"support_candidate_count": 8},
        "live_smoke": {
            "openai_total_tokens": 1943,
            "support_candidate_count": 8,
        },
        "auth_smoke": {"auth_used": True},
    }


def _audit_manifest(*, raw_sensitive_text: bool = False) -> dict:
    sensitive = [{"field": "story", "sha256": "a" * 64}]
    if raw_sensitive_text:
        sensitive[0]["text"] = "private narrative"
    return {
        "status": "pass",
        "summary": {
            "input_summary_count": 5,
            "found_artifact_count": 41,
            "missing_reference_count": 0,
            "total_artifact_bytes": 54239899,
            "sensitive_text_count": 3,
        },
        "sensitive_text": sensitive,
    }


def _browser_report(status: str = "pass") -> dict:
    return {
        "status": status,
        "screenshots": [
            {
                "label": "desktop",
                "status": "pass",
                "width": 1440,
                "height": 1200,
                "bytes": 123,
            },
            {
                "label": "mobile",
                "status": status,
                "width": 390,
                "height": 900,
                "bytes": 456,
            },
        ],
    }


def _prefix_run_record(status: str = "pass") -> dict:
    return {
        "status": status,
        "record_id": "abc123",
        "record_type": "prefix_latent_narrative_scenario_run",
        "condition": {
            "source": "external_condition_report",
            "text_memory_dim": 128,
            "market_implication_count": 1,
            "forward_warning_count": 1,
            "narrative_text_sha256": "a" * 64,
        },
        "support": {"candidate_count": 8},
        "validation": {
            "overall_status": "pass",
            "selected_start_status": status,
        },
        "artifacts": [
            {"key": "report", "path": "report.json", "sha256": "b" * 64},
            {"key": "markdown", "path": "report.md", "sha256": "c" * 64},
        ],
    }


def test_parse_evidence_spec_requires_name_type_path() -> None:
    assert parse_evidence_spec("qa:qa_packet:report.json") == {
        "name": "qa",
        "type": "qa_packet",
        "path": "report.json",
    }

    try:
        parse_evidence_spec("bad")
    except ValueError as error:
        assert "name:type:path" in str(error)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("expected invalid spec to fail")


def test_browser_qa_summary_counts_screenshots() -> None:
    summary = browser_qa_summary(_browser_report())

    assert summary["status"] == "pass"
    assert summary["screenshot_count"] == 2
    assert summary["pass_screenshot_count"] == 2
    assert summary["viewports"][1]["label"] == "mobile"


def test_prefix_run_record_summary_counts_support_and_hashes() -> None:
    summary = prefix_run_record_summary(_prefix_run_record())

    assert summary["record_type"] == "prefix_latent_narrative_scenario_run"
    assert summary["text_memory_dim"] == 128
    assert summary["support_candidate_count"] == 8
    assert summary["narrative_hash_present"] is True
    assert summary["hashed_artifact_count"] == 2


def test_registry_gates_pass_for_clean_evidence() -> None:
    entries = [
        {"type": "qa_packet", "summary": {"status": "pass", "failed_gates": [], "pass_gate_count": 8, "gate_count": 8}},
        {
            "type": "audit_manifest",
            "summary": {
                "status": "pass",
                "missing_reference_count": 0,
                "raw_sensitive_text_retained": False,
                "found_artifact_count": 41,
            },
        },
        {
            "type": "browser_qa",
            "summary": {
                "status": "pass",
                "screenshot_count": 2,
                "pass_screenshot_count": 2,
            },
        },
        {
            "type": "prefix_run_record",
            "summary": prefix_run_record_summary(_prefix_run_record()),
        },
    ]

    gates = build_registry_gates(entries)

    assert overall_status(gates) == "pass"
    assert {gate["name"] for gate in gates} == {
        "qa_packet_pass",
        "audit_manifest_pass",
        "browser_render_pass",
        "prefix_run_record_pass",
    }


def test_registry_gates_fail_on_raw_sensitive_text_or_bad_browser() -> None:
    entries = [
        {"type": "qa_packet", "summary": {"status": "pass", "failed_gates": [], "pass_gate_count": 8, "gate_count": 8}},
        {
            "type": "audit_manifest",
            "summary": {
                "status": "pass",
                "missing_reference_count": 0,
                "raw_sensitive_text_retained": True,
                "found_artifact_count": 41,
            },
        },
        {
            "type": "browser_qa",
            "summary": {
                "status": "fail",
                "screenshot_count": 2,
                "pass_screenshot_count": 1,
            },
        },
        {
            "type": "prefix_run_record",
            "summary": prefix_run_record_summary(_prefix_run_record(status="fail")),
        },
    ]

    by_name = {gate["name"]: gate for gate in build_registry_gates(entries)}

    assert by_name["audit_manifest_pass"]["status"] == "fail"
    assert by_name["browser_render_pass"]["status"] == "fail"
    assert by_name["prefix_run_record_pass"]["status"] == "fail"


def test_build_registry_writes_safe_summary_without_raw_payloads(tmp_path: Path) -> None:
    qa = tmp_path / "qa.json"
    audit = tmp_path / "audit.json"
    browser = tmp_path / "browser.json"
    run_record = tmp_path / "run_record.json"
    qa.write_text(json.dumps(_qa_packet()), encoding="utf-8")
    audit.write_text(json.dumps(_audit_manifest()), encoding="utf-8")
    browser.write_text(json.dumps(_browser_report()), encoding="utf-8")
    run_record.write_text(json.dumps(_prefix_run_record()), encoding="utf-8")

    registry = build_registry(
        SimpleNamespace(
            repo_root=".",
            output_dir=str(tmp_path / "registry"),
            evidence=[
                f"qa_packet:qa_packet:{qa}",
                f"audit_manifest:audit_manifest:{audit}",
                f"browser_qa:browser_qa:{browser}",
                f"prefix_run_record:prefix_run_record:{run_record}",
            ],
        )
    )

    assert registry["status"] == "pass"
    assert registry["summary"]["evidence_count"] == 4
    assert all("sha256" in row for row in registry["evidence"])
    serialized = json.dumps(registry)
    assert "private narrative" not in serialized
    assert (tmp_path / "registry" / "demo_run_registry.json").is_file()
    assert (tmp_path / "registry" / "demo_run_registry.md").is_file()


def test_build_registry_fails_when_required_evidence_is_missing(tmp_path: Path) -> None:
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps(_qa_packet()), encoding="utf-8")

    registry = build_registry(
        SimpleNamespace(
            repo_root=".",
            output_dir=str(tmp_path / "registry"),
            evidence=[
                f"qa_packet:qa_packet:{qa}",
                f"audit_manifest:audit_manifest:{tmp_path / 'missing.json'}",
                f"browser_qa:browser_qa:{tmp_path / 'missing_browser.json'}",
                f"prefix_run_record:prefix_run_record:{tmp_path / 'missing_record.json'}",
            ],
        )
    )

    assert registry["status"] == "fail"
    assert registry["summary"]["missing_count"] == 3
    assert any(gate["name"] == "required_evidence_present" for gate in registry["gates"])


def test_render_markdown_explains_registry_contract() -> None:
    registry = {
        "status": "pass",
        "source_commit": "abc123",
        "summary": {"evidence_count": 1, "existing_count": 1, "missing_count": 0},
        "gates": [
            {"name": "qa_packet_pass", "status": "pass", "evidence": "ok"},
        ],
        "evidence": [
            {
                "name": "qa",
                "type": "qa_packet",
                "summary": {"status": "pass"},
                "bytes": 10,
                "sha256": "a" * 64,
                "path": "qa.json",
            }
        ],
    }

    markdown = render_markdown(registry)

    assert "Narrative Demo Run Registry" in markdown
    assert "Source commit: `abc123`" in markdown
    assert "does not copy raw report payloads" in markdown
