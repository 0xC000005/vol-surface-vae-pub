import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_demo_audit_manifest import (
    build_audit_manifest,
    iter_path_strings,
    iter_sensitive_text,
    render_markdown,
    resolve_artifact_path,
    summarize_sensitive_text,
)


def test_iter_path_strings_finds_artifact_like_fields() -> None:
    payload = {
        "prefix_report_path": "outputs/report.json",
        "artifact_paths": {"markdown": "outputs/report.md"},
        "not_path": "hello world",
    }

    rows = list(iter_path_strings(payload))

    assert ("prefix_report_path", "outputs/report.json") in rows
    assert ("artifact_paths.markdown", "outputs/report.md") in rows
    assert all(value != "hello world" for _, value in rows)


def test_resolve_artifact_path_checks_roots(tmp_path: Path) -> None:
    artifact = tmp_path / "outputs" / "report.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")

    assert resolve_artifact_path("outputs/report.json", [tmp_path]) == artifact
    assert resolve_artifact_path("missing.json", [tmp_path]) is None


def test_sensitive_text_is_hashed_by_default() -> None:
    payload = {
        "story": "A private risk-manager narrative.",
        "history_start_date": "2016-04-27",
    }
    summary_path = Path("summary.json")

    rows = summarize_sensitive_text(
        [(summary_path, payload)],
        include_text=False,
    )

    assert rows[0]["field"] == "story"
    assert rows[0]["characters"] == len(payload["story"])
    assert "text" not in rows[0]
    assert len(rows[0]["sha256"]) == 64
    assert list(iter_sensitive_text(payload))[0][1] == payload["story"]
    assert len(rows) == 1


def test_build_audit_manifest_hashes_and_optionally_copies_artifacts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = repo / "outputs"
    output.mkdir()
    report = output / "report.json"
    markdown = output / "report.md"
    report.write_text('{"ok": true}\n', encoding="utf-8")
    markdown.write_text("# Report\n", encoding="utf-8")
    summary = repo / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "story": "A private narrative.",
                "prefix_report_path": "outputs/report.json",
                "artifact_paths": {"markdown": "outputs/report.md"},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_audit_manifest(
        SimpleNamespace(
            repo_root=str(repo),
            run_summary=[str(summary)],
            artifact_root=[],
            output_dir=str(repo / "audit"),
            copy_artifacts=True,
            include_sensitive_text=False,
        )
    )

    assert manifest["status"] == "pass"
    assert manifest["summary"]["found_artifact_count"] == 2
    assert manifest["summary"]["sensitive_text_count"] == 1
    assert all("archived_path" in row for row in manifest["artifacts"])
    assert (repo / "audit" / "demo_audit_manifest.json").is_file()
    assert (repo / "audit" / "demo_audit_manifest.md").is_file()
    assert Path(manifest["artifacts"][0]["archived_path"]).is_file()


def test_build_audit_manifest_marks_missing_artifact_reference_as_fail(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    summary = repo / "summary.json"
    summary.write_text(
        json.dumps({"prefix_report_path": "outputs/missing.json"}),
        encoding="utf-8",
    )

    manifest = build_audit_manifest(
        SimpleNamespace(
            repo_root=str(repo),
            run_summary=[str(summary)],
            artifact_root=[],
            output_dir=str(repo / "audit"),
            copy_artifacts=False,
            include_sensitive_text=False,
        )
    )

    assert manifest["status"] == "fail"
    assert manifest["summary"]["blocking_missing_reference_count"] == 1
    assert manifest["missing_references"][0]["path"] == "outputs/missing.json"


def test_render_markdown_summarizes_manifest() -> None:
    manifest = {
        "status": "pass",
        "source_commit": "abc123",
        "summary": {
            "input_summary_count": 1,
            "found_artifact_count": 1,
            "missing_reference_count": 0,
            "total_artifact_bytes": 10,
            "sensitive_text_count": 1,
            "copy_artifacts": False,
        },
        "artifacts": [{"bytes": 10, "sha256": "a" * 64, "path": "report.json"}],
        "missing_references": [],
    }

    markdown = render_markdown(manifest)

    assert "Narrative Demo Audit Manifest" in markdown
    assert "Source commit: `abc123`" in markdown
    assert "Sensitive Text Handling" in markdown
