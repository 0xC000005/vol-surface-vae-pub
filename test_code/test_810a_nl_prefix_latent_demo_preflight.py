import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_demo_preflight import (
    DEFAULT_REQUIRED_ARTIFACTS,
    build_preflight_report,
    render_preflight_markdown,
    write_preflight_outputs,
)


def test_build_preflight_report_hashes_required_artifacts(tmp_path: Path) -> None:
    (tmp_path / "bundle").mkdir()
    first = tmp_path / "bundle" / "a.txt"
    second = tmp_path / "bundle" / "b.bin"
    first.write_text("alpha", encoding="utf-8")
    second.write_bytes(b"beta")

    report = build_preflight_report(
        repo_root=tmp_path,
        artifact_paths=["bundle/a.txt", "bundle/b.bin"],
        status_lines=[],
        openai_key_env={},
        check_env_file_ignored=False,
    )

    assert report["status"] == "pass"
    assert report["summary"]["required_artifact_count"] == 2
    assert report["summary"]["missing_artifact_count"] == 0
    assert report["summary"]["total_required_bytes"] == 9
    assert [item["path"] for item in report["artifacts"]["required"]] == [
        "bundle/a.txt",
        "bundle/b.bin",
    ]
    assert all(len(item["sha256"]) == 64 for item in report["artifacts"]["required"])


def test_preflight_flags_missing_artifacts_unsafe_staging_and_secret_presence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")

    report = build_preflight_report(
        repo_root=tmp_path,
        artifact_paths=[
            "models/backfill/demo/best_model.pt",
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/run/report.json",
        ],
        status_lines=[
            "A  experiments/backfill/block_ar/nl_scenario_demo_outputs/run/report.json",
            "A  .env",
            " M docs/research_protocols/demo.md",
        ],
        openai_key_env={"OPENAI_API_KEY": "sk-test-secret-value"},
        check_env_file_ignored=False,
    )

    encoded = json.dumps(report)
    assert "sk-test-secret-value" not in encoded
    assert report["status"] == "fail"
    assert report["summary"]["missing_artifact_count"] == 2
    assert report["secrets"]["openai_api_key_present"] is True
    assert report["git_hygiene"]["unsafe_staged_paths"] == [
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/run/report.json",
        ".env",
    ]


def test_render_preflight_markdown_surfaces_next_actions(tmp_path: Path) -> None:
    report = build_preflight_report(
        repo_root=tmp_path,
        artifact_paths=["missing.pt"],
        status_lines=["A  .env"],
        openai_key_env={},
        check_env_file_ignored=False,
    )

    markdown = render_preflight_markdown(report)

    assert "Status: `fail`" in markdown
    assert "missing.pt" in markdown
    assert ".env" in markdown
    assert "OPENAI key present: `False`" in markdown


def test_write_preflight_outputs_persists_artifact_paths(tmp_path: Path) -> None:
    report = build_preflight_report(
        repo_root=tmp_path,
        artifact_paths=[],
        status_lines=[],
        openai_key_env={},
        check_env_file_ignored=False,
    )

    paths = write_preflight_outputs(report, tmp_path / "out")
    saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

    assert saved["artifact_paths"] == paths
    assert Path(paths["markdown"]).is_file()


def test_default_required_artifacts_include_cached_casebook_condition_reports() -> None:
    required = set(DEFAULT_REQUIRED_ARTIFACTS)

    for folder in [
        "prefix_latent_condition_only_report_822a_commodity",
        "prefix_latent_condition_only_report_823a_dollar",
        "prefix_latent_condition_only_report_823b_safe_haven",
    ]:
        base = f"experiments/backfill/block_ar/nl_scenario_demo_outputs/{folder}"
        assert f"{base}/condition_only_report.json" in required
        assert f"{base}/condition_only_report_arrays.npz" in required


def test_default_required_artifacts_include_base_data_files() -> None:
    required = set(DEFAULT_REQUIRED_ARTIFACTS)

    assert "data/vol_surface_with_ret.npz" in required
    assert "data/spx_vol_surface_history_full_data_fixed.parquet" in required
    assert "data/multi_factor_levels.parquet" in required
    assert "data/multi_factor_returns.parquet" in required
