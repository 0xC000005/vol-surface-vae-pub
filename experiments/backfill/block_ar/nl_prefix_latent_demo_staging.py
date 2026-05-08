"""Create a clean staging tree for the narrative prefix-latent demo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.backfill.block_ar.nl_prefix_latent_demo_preflight import (
    DEFAULT_REQUIRED_ARTIFACTS,
    sha256_file,
)


EXCLUDED_SOURCE_EXACT = {".env"}
EXCLUDED_SOURCE_PREFIXES = (
    ".agents/",
    ".claude/",
    ".venv/",
    "data/",
    "models/",
    "paper/",
    "results/",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/",
    "autoresearch-session/",
)
EXCLUDED_SOURCE_SUFFIXES = (".pdf",)


def _normalize(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def should_copy_source_file(path: str) -> bool:
    relative = _normalize(path)
    if not relative or relative in EXCLUDED_SOURCE_EXACT:
        return False
    if relative.lower().endswith(EXCLUDED_SOURCE_SUFFIXES):
        return False
    return not any(relative.startswith(prefix) for prefix in EXCLUDED_SOURCE_PREFIXES)


def git_tracked_files(source_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(source_root), "ls-files"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [_normalize(line) for line in proc.stdout.splitlines() if line.strip()]


def _copy_file(source_root: Path, stage_root: Path, relative: str) -> dict[str, Any]:
    source = source_root / relative
    target = stage_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    size = target.stat().st_size
    return {
        "path": relative,
        "bytes": int(size),
        "sha256": sha256_file(target),
    }


def _fail_manifest(
    *,
    source_root: Path,
    stage_root: Path,
    errors: Sequence[str],
    missing_artifacts: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "status": "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "stage_root": str(stage_root),
        "errors": list(errors),
        "missing_artifacts": list(missing_artifacts),
        "summary": {
            "copied_source_count": 0,
            "copied_artifact_count": 0,
            "total_source_bytes": 0,
            "total_artifact_bytes": 0,
        },
    }


def stage_demo_bundle(
    *,
    source_root: str | Path = ".",
    stage_root: str | Path,
    tracked_files: Sequence[str] | None = None,
    artifact_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    stage = Path(stage_root).resolve()
    artifacts = [_normalize(path) for path in (artifact_paths or DEFAULT_REQUIRED_ARTIFACTS)]

    if stage.exists() and any(stage.iterdir()):
        return _fail_manifest(
            source_root=source,
            stage_root=stage,
            errors=["stage_root_not_empty"],
        )

    missing = [path for path in artifacts if not (source / path).is_file()]
    if missing:
        return _fail_manifest(
            source_root=source,
            stage_root=stage,
            errors=["missing_required_artifacts"],
            missing_artifacts=missing,
        )

    stage.mkdir(parents=True, exist_ok=True)
    source_files = [
        _normalize(path)
        for path in (git_tracked_files(source) if tracked_files is None else tracked_files)
    ]
    copied_source = []
    for relative in source_files:
        if not should_copy_source_file(relative):
            continue
        if not (source / relative).is_file():
            continue
        copied_source.append(_copy_file(source, stage, relative))

    copied_artifacts = [_copy_file(source, stage, relative) for relative in artifacts]
    manifest = {
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "stage_root": str(stage),
        "errors": [],
        "missing_artifacts": [],
        "excluded_source_policy": {
            "exact": sorted(EXCLUDED_SOURCE_EXACT),
            "prefixes": list(EXCLUDED_SOURCE_PREFIXES),
            "suffixes": list(EXCLUDED_SOURCE_SUFFIXES),
        },
        "summary": {
            "copied_source_count": len(copied_source),
            "copied_artifact_count": len(copied_artifacts),
            "total_source_bytes": int(sum(item["bytes"] for item in copied_source)),
            "total_artifact_bytes": int(sum(item["bytes"] for item in copied_artifacts)),
        },
        "copied_source": copied_source,
        "copied_artifacts": copied_artifacts,
        "artifact_paths": {"manifest": str(stage / "demo_staging_manifest.json")},
    }
    (stage / "demo_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def run_staged_preflight(stage_root: str | Path, output_dir: str) -> dict[str, Any]:
    stage = Path(stage_root).resolve()
    env = os.environ.copy()
    env.setdefault(
        "UV_PROJECT_ENVIRONMENT",
        str(stage.parent / f"{stage.name}_uv_env"),
    )
    command = [
        "uv",
        "run",
        "python",
        "experiments/backfill/block_ar/nl_prefix_latent_demo_preflight.py",
        "--repo-root",
        ".",
        "--output-dir",
        output_dir,
    ]
    proc = subprocess.run(
        command,
        cwd=stage,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--stage-root", required=True)
    parser.add_argument(
        "--artifact-path",
        dest="artifact_paths",
        action="append",
        help="Required artifact path. Repeat to override the default bundle.",
    )
    parser.add_argument("--run-preflight", action="store_true")
    parser.add_argument(
        "--preflight-output-dir",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_demo_preflight_staged"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = stage_demo_bundle(
        source_root=args.source_root,
        stage_root=args.stage_root,
        artifact_paths=args.artifact_paths,
    )
    if manifest["status"] == "pass" and args.run_preflight:
        preflight = run_staged_preflight(args.stage_root, args.preflight_output_dir)
        manifest["preflight"] = preflight
        manifest["status"] = "pass" if preflight["status"] == "pass" else "fail"
        Path(args.stage_root, "demo_staging_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "stage_root": manifest["stage_root"],
                "manifest": str(Path(args.stage_root) / "demo_staging_manifest.json"),
            },
            sort_keys=True,
        )
    )
    if manifest["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
