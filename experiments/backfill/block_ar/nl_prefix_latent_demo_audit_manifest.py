#!/usr/bin/env python
"""Create an audit manifest for narrative demo run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_demo_audit_manifest_838a"
)
PATH_HINTS = ("path", "report", "markdown", "arrays", "summary", "manifest")
ARTIFACT_SUFFIXES = (".json", ".md", ".npz", ".pt", ".jsonl", ".parquet")
SENSITIVE_TEXT_KEYS = ("story", "narrative", "prompt")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _looks_like_artifact_path(key: str, value: str) -> bool:
    normalized = value.replace("\\", "/").strip()
    if not normalized or "\n" in normalized:
        return False
    key_lower = key.lower()
    if any(hint in key_lower for hint in PATH_HINTS):
        return (
            normalized.endswith(ARTIFACT_SUFFIXES)
            or "/" in normalized
            or normalized.startswith("/tmp/")
        )
    return normalized.endswith(ARTIFACT_SUFFIXES)


def iter_path_strings(payload: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_path_strings(value, child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child = f"{prefix}[{index}]"
            yield from iter_path_strings(value, child)
    elif isinstance(payload, str) and _looks_like_artifact_path(prefix, payload):
        yield prefix, payload


def iter_sensitive_text(payload: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_sensitive_text(value, child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child = f"{prefix}[{index}]"
            yield from iter_sensitive_text(value, child)
    elif isinstance(payload, str):
        if _is_sensitive_text_key(prefix):
            yield prefix, payload


def _is_sensitive_text_key(key: str) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", key.lower()) if token]
    return any(token in SENSITIVE_TEXT_KEYS for token in tokens)


def _candidate_roots(repo_root: Path, summary_paths: Sequence[Path]) -> list[Path]:
    roots = [repo_root]
    for path in summary_paths:
        roots.append(path.parent)
        for parent in path.parents:
            if parent.name in {"vol-surface-vae-pub", "nl_prefix_latent_demo_stage_832a", "nl_prefix_latent_demo_stage_837a_auth"}:
                roots.append(parent)
                break
    unique = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_artifact_path(raw_path: str, roots: Sequence[Path]) -> Path | None:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for root in roots:
        joined = root / raw_path
        if joined.is_file():
            return joined
    return None


def _safe_archive_relative(path: Path) -> Path:
    parts = [part for part in path.parts if part not in {"", "/"}]
    if path.is_absolute():
        return Path("absolute").joinpath(*parts)
    return path


def summarize_artifacts(
    *,
    summaries: Sequence[tuple[Path, Mapping[str, Any]]],
    roots: Sequence[Path],
    output_dir: Path,
    copy_artifacts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    found: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    archive_root = output_dir / "artifacts"
    for summary_path, payload in summaries:
        for key, raw_path in iter_path_strings(payload):
            resolved = resolve_artifact_path(raw_path, roots)
            if resolved is None:
                missing.append(
                    {
                        "summary": str(summary_path),
                        "field": key,
                        "path": raw_path,
                    }
                )
                continue
            resolved_key = str(resolved)
            if resolved_key in found:
                found[resolved_key]["referenced_by"].append(
                    {"summary": str(summary_path), "field": key}
                )
                continue
            record = {
                "path": raw_path,
                "resolved_path": resolved_key,
                "bytes": int(resolved.stat().st_size),
                "sha256": sha256_file(resolved),
                "referenced_by": [{"summary": str(summary_path), "field": key}],
            }
            if copy_artifacts:
                archive_path = archive_root / _safe_archive_relative(Path(raw_path))
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved, archive_path)
                record["archived_path"] = str(archive_path)
            found[resolved_key] = record
    return list(found.values()), missing


def summarize_sensitive_text(
    summaries: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    include_text: bool,
) -> list[dict[str, Any]]:
    rows = []
    for summary_path, payload in summaries:
        for key, value in iter_sensitive_text(payload):
            row = {
                "summary": str(summary_path),
                "field": key,
                "characters": len(value),
                "sha256": sha256_text(value),
            }
            if include_text:
                row["text"] = value
            rows.append(row)
    return rows


def render_markdown(manifest: Mapping[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        "# Narrative Demo Audit Manifest",
        "",
        f"Status: `{manifest['status']}`",
        "",
        "## Scope",
        "",
        (
            "This manifest records hashes and optional archive copies for saved "
            "demo evidence. It is intended as a durable audit handoff, not a "
            "source-code artifact."
        ),
        "",
        "## Summary",
        "",
        f"- Source commit: `{manifest.get('source_commit', '')}`",
        f"- Input summaries: `{summary['input_summary_count']}`",
        f"- Found artifacts: `{summary['found_artifact_count']}`",
        f"- Missing references: `{summary['missing_reference_count']}`",
        f"- Total artifact bytes: `{summary['total_artifact_bytes']}`",
        f"- Sensitive text entries: `{summary['sensitive_text_count']}`",
        f"- Artifact copies written: `{summary['copy_artifacts']}`",
        "",
        "## Artifacts",
        "",
        "| Bytes | SHA-256 | Path |",
        "| ---: | --- | --- |",
    ]
    for row in manifest["artifacts"]:
        lines.append(
            f"| {row['bytes']} | `{row['sha256'][:12]}` | "
            f"`{row['path']}` |"
        )
    if manifest["missing_references"]:
        lines.extend(["", "## Missing References", "", "| Field | Path |", "| --- | --- |"])
        for row in manifest["missing_references"]:
            lines.append(f"| `{row['field']}` | `{row['path']}` |")
    lines.extend(
        [
            "",
            "## Sensitive Text Handling",
            "",
            (
                "Narrative-like text fields are hashed by default. Use "
                "`--include-sensitive-text` only for an internal audit store "
                "that is approved to retain user narratives."
            ),
        ]
    )
    return "\n".join(lines)


def build_audit_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = [Path(path) for path in args.run_summary]
    summaries = [(path, _load_json(path)) for path in summary_paths]
    roots = _candidate_roots(repo_root, summary_paths)
    roots.extend(Path(path).resolve() for path in getattr(args, "artifact_root", []))

    artifacts, missing = summarize_artifacts(
        summaries=summaries,
        roots=roots,
        output_dir=output_dir,
        copy_artifacts=bool(args.copy_artifacts),
    )
    sensitive_text = summarize_sensitive_text(
        summaries,
        include_text=bool(args.include_sensitive_text),
    )
    total_bytes = sum(int(row["bytes"]) for row in artifacts)
    blocking_missing = [
        row
        for row in missing
        if Path(str(row.get("path", ""))).suffix in ARTIFACT_SUFFIXES
    ]
    manifest = {
        "status": "fail" if blocking_missing else "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_head(repo_root),
        "inputs": {
            "run_summaries": [str(path) for path in summary_paths],
            "artifact_roots": [str(root) for root in roots],
        },
        "summary": {
            "input_summary_count": len(summaries),
            "found_artifact_count": len(artifacts),
            "missing_reference_count": len(missing),
            "blocking_missing_reference_count": len(blocking_missing),
            "total_artifact_bytes": int(total_bytes),
            "sensitive_text_count": len(sensitive_text),
            "copy_artifacts": bool(args.copy_artifacts),
        },
        "artifacts": artifacts,
        "missing_references": missing,
        "sensitive_text": sensitive_text,
        "artifact_paths": {
            "json": str(output_dir / "demo_audit_manifest.json"),
            "markdown": str(output_dir / "demo_audit_manifest.md"),
        },
    }
    _write_json(manifest["artifact_paths"]["json"], manifest)
    Path(manifest["artifact_paths"]["markdown"]).write_text(
        render_markdown(manifest).rstrip() + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-summary", action="append", required=True)
    parser.add_argument("--artifact-root", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copy-artifacts", action="store_true")
    parser.add_argument("--include-sensitive-text", action="store_true")
    args = parser.parse_args()
    manifest = build_audit_manifest(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "artifact_count": manifest["summary"]["found_artifact_count"],
                "missing": manifest["summary"]["missing_reference_count"],
                "json": manifest["artifact_paths"]["json"],
                "markdown": manifest["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
