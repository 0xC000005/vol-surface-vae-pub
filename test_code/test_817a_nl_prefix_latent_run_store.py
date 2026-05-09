import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_run_store import (
    build_store,
    connect_store,
    store_summary,
    upsert_registry,
    upsert_run_record,
)


def _run_record() -> dict:
    return {
        "record_id": "run_001",
        "created_at": "2026-05-08T00:00:00+00:00",
        "source_commit": "abc123",
        "status": "pass",
        "condition": {
            "source": "external_condition_report",
            "narrative_text_sha256": "a" * 64,
        },
        "validation": {"selected_start_status": "pass"},
        "support": {"candidate_count": 8},
        "artifacts": [
            {
                "key": "report",
                "path": "report.json",
                "exists": True,
                "bytes": 10,
                "sha256": "b" * 64,
            }
        ],
    }


def _registry() -> dict:
    return {
        "evidence": [
            {
                "name": "prefix_run_record",
                "type": "prefix_run_record",
                "path": "run_record.json",
                "bytes": 20,
                "sha256": "c" * 64,
                "summary": {"status": "pass", "support_candidate_count": 8},
            }
        ],
        "gates": [
            {
                "name": "prefix_run_record_pass",
                "status": "pass",
                "evidence": "support=8",
            }
        ],
    }


def test_upsert_run_record_and_registry_persist_rows(tmp_path: Path) -> None:
    conn = connect_store(tmp_path / "store.sqlite")
    try:
        upsert_run_record(conn, _run_record())
        upsert_registry(conn, _registry())
        conn.commit()
        summary = store_summary(conn)
    finally:
        conn.close()

    assert summary["run_record_count"] == 1
    assert summary["run_artifact_count"] == 1
    assert summary["registry_evidence_count"] == 1
    assert summary["registry_gate_count"] == 1
    assert summary["run_status_counts"] == {"pass": 1}


def test_upsert_run_record_is_idempotent(tmp_path: Path) -> None:
    conn = connect_store(tmp_path / "store.sqlite")
    try:
        record = _run_record()
        upsert_run_record(conn, record)
        record["support"]["candidate_count"] = 9
        upsert_run_record(conn, record)
        conn.commit()
        count, support = conn.execute(
            "SELECT COUNT(*), support_candidate_count FROM run_records"
        ).fetchone()
    finally:
        conn.close()

    assert count == 1
    assert support == 9


def test_build_store_writes_summary_json(tmp_path: Path) -> None:
    run_record = tmp_path / "run_record.json"
    registry = tmp_path / "registry.json"
    run_record.write_text(json.dumps(_run_record()), encoding="utf-8")
    registry.write_text(json.dumps(_registry()), encoding="utf-8")

    summary = build_store(
        SimpleNamespace(
            sqlite=str(tmp_path / "store.sqlite"),
            summary_json=str(tmp_path / "summary.json"),
            run_record=[str(run_record)],
            registry=[str(registry)],
        )
    )

    assert summary["run_record_count"] == 1
    assert summary["registry_gate_count"] == 1
    assert (tmp_path / "summary.json").is_file()
    with sqlite3.connect(tmp_path / "store.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_records").fetchone()[0] == 1
