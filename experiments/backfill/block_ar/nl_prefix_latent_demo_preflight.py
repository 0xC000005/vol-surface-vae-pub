"""Preflight checks for the narrative prefix-latent Gradio demo bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_REQUIRED_ARTIFACTS = [
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/args.json",
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt",
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/train_summary.json",
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/training_history.json",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_openai_schema_v2_representative_220/narrative_adapter.pt",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_openai_schema_v2_representative_220/narrative_label_cache.jsonl",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_openai_schema_v2_representative_220/narrative_pipeline_arrays.npz",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_openai_schema_v2_representative_220/narrative_pipeline_report.json",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_bridge_eval_openai_schema_v2_representative_220/bridge_adapter.pt",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_arrays.npz",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_boss_demo_pack_829a_live_casebook/boss_demo_pack.json",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_boss_demo_pack_829a_live_casebook/boss_demo_pack.md",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_api_casebook_828a_three_story/gradio_live_api_casebook_summary.json",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_api_casebook_828a_three_story/gradio_live_api_casebook_summary.md",
]

UNSAFE_STAGED_EXACT = {".env"}
UNSAFE_STAGED_PREFIXES = (
    "data/",
    "models/",
    "paper/",
    "results/",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/",
)


def _run_git(repo_root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def git_status_lines(repo_root: Path) -> list[str]:
    proc = _run_git(repo_root, ["status", "--porcelain"])
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def git_check_ignored(repo_root: Path, relative_path: str) -> bool | None:
    proc = _run_git(repo_root, ["check-ignore", "-q", "--", relative_path])
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _status_path(line: str) -> tuple[str, str]:
    if len(line) < 4:
        return line[:2], ""
    code = line[:2]
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()
    return code, _normalize_relative_path(path)


def _is_staged(code: str) -> bool:
    return bool(code) and code[0] not in {" ", "?"}


def _is_unsafe_staged_path(path: str) -> bool:
    if path in UNSAFE_STAGED_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in UNSAFE_STAGED_PREFIXES)


def inspect_git_hygiene(status_lines: Sequence[str]) -> dict[str, Any]:
    unsafe = []
    staged = []
    for line in status_lines:
        code, path = _status_path(line)
        if _is_staged(code):
            staged.append(path)
            if _is_unsafe_staged_path(path):
                unsafe.append(path)
    return {
        "staged_paths": staged,
        "unsafe_staged_paths": unsafe,
        "unsafe_staged_count": len(unsafe),
    }


def inspect_artifacts(repo_root: Path, artifact_paths: Sequence[str]) -> dict[str, Any]:
    required = []
    missing = []
    total_bytes = 0
    for raw_path in artifact_paths:
        relative = _normalize_relative_path(raw_path)
        path = repo_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        size = path.stat().st_size
        total_bytes += size
        required.append(
            {
                "path": relative,
                "bytes": int(size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "required": required,
        "missing": missing,
        "total_required_bytes": int(total_bytes),
    }


def inspect_secrets(
    repo_root: Path,
    env: Mapping[str, str] | None = None,
    *,
    check_env_file_ignored: bool = True,
) -> dict[str, Any]:
    env_map = os.environ if env is None else env
    env_file = repo_root / ".env"
    ignored: bool | None = None
    if check_env_file_ignored:
        ignored = git_check_ignored(repo_root, ".env")
    return {
        "openai_api_key_present": bool(env_map.get("OPENAI_API_KEY")),
        "env_file_exists": env_file.exists(),
        "env_file_git_ignored": ignored,
    }


def _compute_status(
    *,
    missing_count: int,
    unsafe_staged_count: int,
    env_file_exists: bool,
    env_file_git_ignored: bool | None,
    require_openai_key: bool,
    openai_key_present: bool,
    smoke_status: str | None,
) -> str:
    failed = missing_count > 0 or unsafe_staged_count > 0
    if env_file_exists and env_file_git_ignored is False:
        failed = True
    if require_openai_key and not openai_key_present:
        failed = True
    if smoke_status == "fail":
        failed = True
    return "fail" if failed else "pass"


def build_preflight_report(
    *,
    repo_root: str | Path = ".",
    artifact_paths: Sequence[str] | None = None,
    status_lines: Sequence[str] | None = None,
    openai_key_env: Mapping[str, str] | None = None,
    check_env_file_ignored: bool = True,
    require_openai_key: bool = False,
    smoke_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    artifacts = inspect_artifacts(root, artifact_paths or DEFAULT_REQUIRED_ARTIFACTS)
    git_hygiene = inspect_git_hygiene(
        git_status_lines(root) if status_lines is None else status_lines
    )
    secrets = inspect_secrets(
        root, openai_key_env, check_env_file_ignored=check_env_file_ignored
    )
    smoke = dict(smoke_result or {"status": "not_run"})
    missing_count = len(artifacts["missing"])
    report = {
        "status": _compute_status(
            missing_count=missing_count,
            unsafe_staged_count=int(git_hygiene["unsafe_staged_count"]),
            env_file_exists=bool(secrets["env_file_exists"]),
            env_file_git_ignored=secrets["env_file_git_ignored"],
            require_openai_key=bool(require_openai_key),
            openai_key_present=bool(secrets["openai_api_key_present"]),
            smoke_status=str(smoke.get("status", "")),
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "summary": {
            "required_artifact_count": len(artifact_paths or DEFAULT_REQUIRED_ARTIFACTS),
            "present_artifact_count": len(artifacts["required"]),
            "missing_artifact_count": missing_count,
            "total_required_bytes": artifacts["total_required_bytes"],
            "unsafe_staged_count": git_hygiene["unsafe_staged_count"],
            "openai_key_required": bool(require_openai_key),
        },
        "artifacts": artifacts,
        "git_hygiene": git_hygiene,
        "secrets": secrets,
        "smoke": smoke,
    }
    return report


def run_cached_gradio_smoke(
    *,
    repo_root: str | Path,
    url: str,
    output_dir: str,
    casebook_choice: str,
    samples: int,
    fan_market: str,
    redraw_market: str,
) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "python",
        "experiments/backfill/block_ar/nl_prefix_latent_gradio_api_smoke.py",
        "--url",
        url,
        "--output-dir",
        output_dir,
        "--casebook-choice",
        casebook_choice,
        "--samples",
        str(samples),
        "--fan-market",
        fan_market,
        "--redraw-market",
        redraw_market,
    ]
    proc = subprocess.run(
        command,
        cwd=Path(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def render_preflight_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    artifacts = report.get("artifacts", {})
    git_hygiene = report.get("git_hygiene", {})
    secrets = report.get("secrets", {})
    smoke = report.get("smoke", {})

    lines = [
        "# Narrative Prefix-Latent Demo Preflight",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        f"- Required artifacts: `{summary.get('required_artifact_count', 0)}`",
        f"- Present artifacts: `{summary.get('present_artifact_count', 0)}`",
        f"- Missing artifacts: `{summary.get('missing_artifact_count', 0)}`",
        f"- Total required bytes: `{summary.get('total_required_bytes', 0)}`",
        f"- Unsafe staged paths: `{summary.get('unsafe_staged_count', 0)}`",
        f"- OPENAI key present: `{secrets.get('openai_api_key_present', False)}`",
        f"- `.env` exists: `{secrets.get('env_file_exists', False)}`",
        f"- `.env` ignored: `{secrets.get('env_file_git_ignored')}`",
        f"- Cached smoke: `{smoke.get('status', 'not_run')}`",
        "",
    ]

    missing = list(artifacts.get("missing", []))
    if missing:
        lines.extend(["## Missing Artifacts", ""])
        lines.extend(f"- `{path}`" for path in missing)
        lines.append("")

    unsafe = list(git_hygiene.get("unsafe_staged_paths", []))
    if unsafe:
        lines.extend(["## Unsafe Staged Paths", ""])
        lines.extend(f"- `{path}`" for path in unsafe)
        lines.append("")

    present = list(artifacts.get("required", []))
    if present:
        lines.extend(["## Present Artifacts", ""])
        for item in present:
            lines.append(
                f"- `{item['path']}`: `{item['bytes']}` bytes, sha256 `{item['sha256']}`"
            )
        lines.append("")

    lines.extend(
        [
            "## Next Actions",
            "",
            "- Add or mount every missing artifact before a hosted demo.",
            "- Unstage ignored generated outputs, checkpoints, paper files, and `.env`.",
            "- Keep `OPENAI_API_KEY` in a local `.env` file or platform secret manager only.",
            "- Run cached Gradio smoke before live OpenAI TestFlight.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_preflight_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "demo_preflight_report.json"
    md_path = output_dir / "demo_preflight_report.md"
    paths = {"json": str(json_path), "markdown": str(md_path)}
    report["artifact_paths"] = paths
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_preflight_markdown(report), encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_demo_preflight"
        ),
    )
    parser.add_argument(
        "--artifact-path",
        dest="artifact_paths",
        action="append",
        help="Required artifact path. Repeat to override the default bundle.",
    )
    parser.add_argument("--require-openai-key", action="store_true")
    parser.add_argument("--skip-env-ignore-check", action="store_true")
    parser.add_argument("--run-cached-smoke", action="store_true")
    parser.add_argument("--gradio-url", default="http://127.0.0.1:7862")
    parser.add_argument("--casebook-choice", default="safe_haven_gold_bid:18")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--fan-market", default="SPX")
    parser.add_argument("--redraw-market", default="IV_ATM_3M")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_result: dict[str, Any] | None = None
    if args.run_cached_smoke:
        smoke_result = run_cached_gradio_smoke(
            repo_root=args.repo_root,
            url=args.gradio_url,
            output_dir=str(Path(args.output_dir) / "cached_gradio_smoke"),
            casebook_choice=args.casebook_choice,
            samples=args.samples,
            fan_market=args.fan_market,
            redraw_market=args.redraw_market,
        )
    report = build_preflight_report(
        repo_root=args.repo_root,
        artifact_paths=args.artifact_paths,
        check_env_file_ignored=not args.skip_env_ignore_check,
        require_openai_key=args.require_openai_key,
        smoke_result=smoke_result,
    )
    write_preflight_outputs(report, args.output_dir)
    print(json.dumps({"status": report["status"], **report["artifact_paths"]}, sort_keys=True))
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
