#!/usr/bin/env python
"""SQLite run store for narrative prefix-latent demo records."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SQLITE_PATH = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_run_store_843a/demo_run_store.sqlite"
)
DEFAULT_SUMMARY_PATH = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_run_store_843a/demo_run_store_summary.json"
)
DEFAULT_REGISTRY = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_demo_run_registry_842a_with_run_record/demo_run_registry.json"
)
DEFAULT_RUN_RECORD = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo/prefix_latent_live_smoke/cached_casebook_run/"
    "prefix_latent_condition_only_report_823b_safe_haven/run_record/"
    "prefix_latent_run_record.json"
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


SCHEMA = """
CREATE TABLE IF NOT EXISTS run_records (
  record_id TEXT PRIMARY KEY,
  created_at TEXT,
  source_commit TEXT,
  status TEXT,
  condition_source TEXT,
  selected_start_status TEXT,
  support_candidate_count INTEGER,
  narrative_text_sha256 TEXT,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_artifacts (
  record_id TEXT NOT NULL,
  artifact_key TEXT NOT NULL,
  path TEXT NOT NULL,
  exists_flag INTEGER NOT NULL,
  bytes INTEGER,
  sha256 TEXT,
  PRIMARY KEY (record_id, artifact_key, path)
);

CREATE TABLE IF NOT EXISTS registry_evidence (
  name TEXT PRIMARY KEY,
  evidence_type TEXT NOT NULL,
  status TEXT,
  path TEXT NOT NULL,
  bytes INTEGER,
  sha256 TEXT,
  summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS registry_gates (
  gate_name TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  evidence TEXT
);
"""


def connect_store(sqlite_path: str | Path) -> sqlite3.Connection:
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def upsert_run_record(conn: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    condition = _as_dict(record.get("condition"))
    validation = _as_dict(record.get("validation"))
    support = _as_dict(record.get("support"))
    conn.execute(
        """
        INSERT INTO run_records (
          record_id, created_at, source_commit, status, condition_source,
          selected_start_status, support_candidate_count, narrative_text_sha256,
          payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_id) DO UPDATE SET
          created_at=excluded.created_at,
          source_commit=excluded.source_commit,
          status=excluded.status,
          condition_source=excluded.condition_source,
          selected_start_status=excluded.selected_start_status,
          support_candidate_count=excluded.support_candidate_count,
          narrative_text_sha256=excluded.narrative_text_sha256,
          payload_json=excluded.payload_json
        """,
        (
            str(record.get("record_id", "")),
            str(record.get("created_at", "")),
            str(record.get("source_commit", "")),
            str(record.get("status", "")),
            str(condition.get("source", "")),
            str(validation.get("selected_start_status", "")),
            _as_int(support.get("candidate_count")),
            str(condition.get("narrative_text_sha256", "")),
            json.dumps(record, sort_keys=True),
        ),
    )
    for row in _as_list(record.get("artifacts")):
        item = _as_dict(row)
        conn.execute(
            """
            INSERT INTO run_artifacts (
              record_id, artifact_key, path, exists_flag, bytes, sha256
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id, artifact_key, path) DO UPDATE SET
              exists_flag=excluded.exists_flag,
              bytes=excluded.bytes,
              sha256=excluded.sha256
            """,
            (
                str(record.get("record_id", "")),
                str(item.get("key", "")),
                str(item.get("path", "")),
                1 if item.get("exists") else 0,
                _as_int(item.get("bytes")),
                str(item.get("sha256", "")),
            ),
        )


def upsert_registry(conn: sqlite3.Connection, registry: Mapping[str, Any]) -> None:
    for row in _as_list(registry.get("evidence")):
        item = _as_dict(row)
        summary = _as_dict(item.get("summary"))
        conn.execute(
            """
            INSERT INTO registry_evidence (
              name, evidence_type, status, path, bytes, sha256, summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              evidence_type=excluded.evidence_type,
              status=excluded.status,
              path=excluded.path,
              bytes=excluded.bytes,
              sha256=excluded.sha256,
              summary_json=excluded.summary_json
            """,
            (
                str(item.get("name", "")),
                str(item.get("type", "")),
                str(summary.get("status", "")),
                str(item.get("path", "")),
                _as_int(item.get("bytes")),
                str(item.get("sha256", "")),
                json.dumps(summary, sort_keys=True),
            ),
        )
    for row in _as_list(registry.get("gates")):
        item = _as_dict(row)
        conn.execute(
            """
            INSERT INTO registry_gates (gate_name, status, evidence)
            VALUES (?, ?, ?)
            ON CONFLICT(gate_name) DO UPDATE SET
              status=excluded.status,
              evidence=excluded.evidence
            """,
            (
                str(item.get("name", "")),
                str(item.get("status", "")),
                str(item.get("evidence", "")),
            ),
        )


def store_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    def scalar(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    status_rows = conn.execute(
        "SELECT status, COUNT(*) FROM run_records GROUP BY status ORDER BY status"
    ).fetchall()
    gate_rows = conn.execute(
        "SELECT status, COUNT(*) FROM registry_gates GROUP BY status ORDER BY status"
    ).fetchall()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_record_count": scalar("SELECT COUNT(*) FROM run_records"),
        "run_artifact_count": scalar("SELECT COUNT(*) FROM run_artifacts"),
        "registry_evidence_count": scalar("SELECT COUNT(*) FROM registry_evidence"),
        "registry_gate_count": scalar("SELECT COUNT(*) FROM registry_gates"),
        "run_status_counts": {str(status): int(count) for status, count in status_rows},
        "gate_status_counts": {str(status): int(count) for status, count in gate_rows},
    }


def build_store(args: argparse.Namespace) -> dict[str, Any]:
    conn = connect_store(args.sqlite)
    try:
        for path in args.run_record:
            upsert_run_record(conn, _load_json(path))
        for path in args.registry:
            upsert_registry(conn, _load_json(path))
        conn.commit()
        summary = store_summary(conn)
    finally:
        conn.close()
    summary["sqlite"] = str(args.sqlite)
    _write_json(args.summary_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--registry", action="append", default=None)
    parser.add_argument("--run-record", action="append", default=None)
    args = parser.parse_args()
    if args.registry is None:
        args.registry = [DEFAULT_REGISTRY]
    if args.run_record is None:
        args.run_record = [DEFAULT_RUN_RECORD]
    summary = build_store(args)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
