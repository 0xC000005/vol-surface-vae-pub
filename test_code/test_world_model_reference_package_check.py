import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.reference_package_check import (
    check_reference_package,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reference_package_check_validates_guardrail_terms(tmp_path):
    report = tmp_path / "reports" / "source.md"
    report.parent.mkdir(parents=True)
    report.write_text("source report\n", encoding="utf-8")

    doc = tmp_path / "docs" / "guardrail.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("alpha\nbeta caveat\n", encoding="utf-8")

    artifact = tmp_path / "artifacts" / "run.json"
    artifact.parent.mkdir(parents=True)
    payload = b'{"ok": true}\n'
    artifact.write_bytes(payload)

    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "source_reports": ["reports/source.md"],
            "guardrail_doc_checks": [
                {"path": "docs/guardrail.md", "required_terms": ["alpha beta"]}
            ],
        },
    )
    digests = tmp_path / "digests.json"
    _write_json(
        digests,
        {
            "entries": [
                {
                    "path": "artifacts/run.json",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ]
        },
    )

    result = check_reference_package(
        root=tmp_path,
        manifest_path=manifest,
        digest_path=digests,
    )

    assert result["ok"] is True
    assert result["checked_reports"] == 1
    assert result["checked_guardrail_docs"] == 1
    assert result["checked_artifacts"] == 1


def test_reference_package_check_reports_missing_guardrail_terms(tmp_path):
    report = tmp_path / "reports" / "source.md"
    report.parent.mkdir(parents=True)
    report.write_text("source report\n", encoding="utf-8")

    doc = tmp_path / "docs" / "guardrail.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("alpha only\n", encoding="utf-8")

    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "source_reports": ["reports/source.md"],
            "guardrail_doc_checks": [
                {"path": "docs/guardrail.md", "required_terms": ["missing caveat"]}
            ],
        },
    )
    digests = tmp_path / "digests.json"
    _write_json(digests, {"entries": []})

    result = check_reference_package(
        root=tmp_path,
        manifest_path=manifest,
        digest_path=digests,
    )

    assert result["ok"] is False
    assert result["guardrail_doc_failures"] == [
        {
            "path": "docs/guardrail.md",
            "reason": "missing_terms",
            "terms": ["missing caveat"],
        }
    ]
