import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_demo_staging import (
    run_staged_preflight,
    stage_demo_bundle,
)


def test_stage_demo_bundle_copies_source_and_required_artifacts_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    (source / "experiments/backfill/block_ar").mkdir(parents=True)
    (source / "experiments/backfill/block_ar/nl_scenario_demo_outputs/run").mkdir(
        parents=True
    )
    (source / "models/backfill/demo").mkdir(parents=True)
    (source / "paper").mkdir()
    (source / "data").mkdir()
    (source / ".venv").mkdir()
    (source / ".agents/skills/private").mkdir(parents=True)
    (source / ".claude").mkdir()
    (source / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (source / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    (source / "paper/private.md").write_text("private paper\n", encoding="utf-8")
    (source / "SSRN23.pdf").write_bytes(b"private paper pdf")
    (source / "data/raw.csv").write_text("x\n", encoding="utf-8")
    (source / ".venv/pyvenv.cfg").write_text("home = /tmp\n", encoding="utf-8")
    (source / ".agents/skills/private/SKILL.md").write_text("secret\n", encoding="utf-8")
    (source / ".claude/settings.json").write_text("{}\n", encoding="utf-8")
    (source / "models/backfill/demo/best_model.pt").write_bytes(b"model")
    generated = (
        source
        / "experiments/backfill/block_ar/nl_scenario_demo_outputs/run/report.json"
    )
    generated.write_text('{"ok": true}\n', encoding="utf-8")

    manifest = stage_demo_bundle(
        source_root=source,
        stage_root=stage,
        tracked_files=[
            "app.py",
            ".env",
            "paper/private.md",
            "SSRN23.pdf",
            "data/raw.csv",
            ".venv/pyvenv.cfg",
            ".agents/skills/private/SKILL.md",
            ".claude/settings.json",
        ],
        artifact_paths=[
            "models/backfill/demo/best_model.pt",
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/run/report.json",
        ],
    )

    assert manifest["status"] == "pass"
    assert (stage / "app.py").is_file()
    assert (stage / "models/backfill/demo/best_model.pt").is_file()
    assert (
        stage / "experiments/backfill/block_ar/nl_scenario_demo_outputs/run/report.json"
    ).is_file()
    assert not (stage / ".env").exists()
    assert not (stage / "paper/private.md").exists()
    assert not (stage / "SSRN23.pdf").exists()
    assert not (stage / "data/raw.csv").exists()
    assert not (stage / ".venv/pyvenv.cfg").exists()
    assert not (stage / ".agents/skills/private/SKILL.md").exists()
    assert not (stage / ".claude/settings.json").exists()
    assert manifest["summary"]["copied_source_count"] == 1
    assert manifest["summary"]["copied_artifact_count"] == 2


def test_stage_demo_bundle_fails_before_copy_when_required_artifact_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    (source / "app.py").write_text("print('ok')\n", encoding="utf-8")

    manifest = stage_demo_bundle(
        source_root=source,
        stage_root=stage,
        tracked_files=["app.py"],
        artifact_paths=["missing/best_model.pt"],
    )

    assert manifest["status"] == "fail"
    assert manifest["missing_artifacts"] == ["missing/best_model.pt"]
    assert not (stage / "app.py").exists()


def test_stage_demo_bundle_refuses_nonempty_stage_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    (stage / "old.txt").write_text("old\n", encoding="utf-8")
    (source / "app.py").write_text("print('ok')\n", encoding="utf-8")

    manifest = stage_demo_bundle(
        source_root=source,
        stage_root=stage,
        tracked_files=["app.py"],
        artifact_paths=[],
    )

    assert manifest["status"] == "fail"
    assert manifest["errors"] == ["stage_root_not_empty"]
    assert not (stage / "app.py").exists()


def test_staging_script_can_be_executed_directly() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/backfill/block_ar/nl_prefix_latent_demo_staging.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert "--stage-root" in proc.stdout


def test_run_staged_preflight_places_uv_environment_outside_stage(
    tmp_path: Path, monkeypatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    calls = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, cwd, check, capture_output, text, env):
        calls.append({"command": command, "cwd": cwd, "env": env})
        return Completed()

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_demo_staging.subprocess.run",
        fake_run,
    )

    result = run_staged_preflight(stage, "outputs")

    uv_env = Path(calls[0]["env"]["UV_PROJECT_ENVIRONMENT"])
    assert result["status"] == "pass"
    assert not uv_env.is_relative_to(stage)
