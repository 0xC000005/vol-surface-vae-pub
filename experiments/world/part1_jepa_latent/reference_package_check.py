from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _resolve_path(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def check_reference_package(
    *,
    root: str | Path = ".",
    manifest_path: str | Path = "experiments/world/part1_jepa_latent/reference_manifest.json",
    digest_path: str | Path = "experiments/world/part1_jepa_latent/reference_artifact_digests.json",
) -> dict[str, Any]:
    root_path = Path(root)
    manifest_file = _resolve_path(root_path, manifest_path)
    digest_file = _resolve_path(root_path, digest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    digests = json.loads(digest_file.read_text(encoding="utf-8"))

    missing_reports = []
    for report_path in manifest.get("source_reports", []):
        resolved = _resolve_path(root_path, report_path)
        if not resolved.exists():
            missing_reports.append(str(report_path))

    artifact_mismatches = []
    checked_artifacts = 0
    for entry in digests.get("entries", []):
        checked_artifacts += 1
        artifact_path = entry["path"]
        resolved = _resolve_path(root_path, artifact_path)
        if not resolved.exists():
            artifact_mismatches.append({"path": artifact_path, "reason": "missing"})
            continue
        payload = resolved.read_bytes()
        actual_bytes = len(payload)
        if actual_bytes != entry.get("bytes"):
            artifact_mismatches.append(
                {
                    "path": artifact_path,
                    "reason": "bytes",
                    "actual": actual_bytes,
                    "expected": entry.get("bytes"),
                }
            )
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != entry.get("sha256"):
            artifact_mismatches.append(
                {
                    "path": artifact_path,
                    "reason": "sha256",
                    "actual": actual_sha,
                    "expected": entry.get("sha256"),
                }
            )

    return {
        "ok": not missing_reports and not artifact_mismatches,
        "manifest_path": str(manifest_file),
        "digest_path": str(digest_file),
        "missing_reports": missing_reports,
        "artifact_mismatches": artifact_mismatches,
        "checked_reports": len(manifest.get("source_reports", [])),
        "checked_artifacts": checked_artifacts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the world-model Part 1 reference package")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/world/part1_jepa_latent/reference_manifest.json"),
    )
    parser.add_argument(
        "--digests",
        type=Path,
        default=Path("experiments/world/part1_jepa_latent/reference_artifact_digests.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = check_reference_package(
        root=args.root,
        manifest_path=args.manifest,
        digest_path=args.digests,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
