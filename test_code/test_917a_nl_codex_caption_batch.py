import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_codex_caption_batch import (
    build_codex_batch_prompt,
    build_codex_prompt,
    run_batch,
    select_bundles,
    strict_caption_batch_schema,
    strict_caption_schema,
)
from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import (
    RiskManagerCaptionV2,
    load_specialist_standards,
)
from test_916a_nl_risk_manager_caption_v2 import _bundle


def test_strict_caption_schema_requires_all_fields() -> None:
    schema = strict_caption_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(RiskManagerCaptionV2.model_json_schema()["properties"])
    assert "training_caption" in schema["required"]
    assert "contrastive_captions" in schema["required"]


def test_select_bundles_supports_split_offset_and_window_ids(tmp_path: Path) -> None:
    bundles = [
        _bundle("joint39_val_0001", split="train", text="train one"),
        _bundle("joint39_val_0002", split="validation", text="val one"),
        _bundle("joint39_val_0003", split="test", text="test one"),
        _bundle("joint39_val_0004", split="test", text="test two"),
    ]
    report_path = tmp_path / "pipeline_report.json"
    report_path.write_text(json.dumps({"narrative_bundles": bundles}), encoding="utf-8")

    selected = select_bundles(report_path, split="test", offset=1, count=1)
    by_id = select_bundles(report_path, window_ids=["joint39_val_0002"])
    balanced = select_bundles(report_path, count=4, selection_mode="split_balanced")

    assert [row["window_id"] for row in selected] == ["joint39_val_0004"]
    assert [row["window_id"] for row in by_id] == ["joint39_val_0002"]
    assert [row["manifest_split"] for row in balanced] == [
        "train",
        "validation",
        "test",
        "test",
    ]


def test_build_codex_prompt_preserves_sidecar_and_leakage_contract() -> None:
    standards = load_specialist_standards()
    prompt = build_codex_prompt(
        _bundle("joint39_val_0001", split="train", text="A broad risk-off tape."),
        standards=standards,
    )

    assert "Return only JSON" in prompt
    assert "Do not call external APIs" in prompt
    assert "training_caption must describe only the current" in prompt
    assert "quant generated scenarios story narrative.docx" in prompt
    assert "SPX: down" in prompt


def test_batch_schema_and_prompt_cover_multiple_windows() -> None:
    schema = strict_caption_batch_schema()
    standards = load_specialist_standards()
    prompt = build_codex_batch_prompt(
        [
            _bundle("joint39_val_0001", split="train", text="A broad risk-off tape."),
            _bundle("joint39_val_0002", split="test", text="A calmer risk-on tape."),
        ],
        standards=standards,
    )

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["captions"]
    assert schema["properties"]["captions"]["type"] == "array"
    assert "Generate one RiskManagerCaptionV2 object per payload" in prompt
    assert "joint39_val_0001" in prompt
    assert "joint39_val_0002" in prompt
    assert prompt.count("quant generated scenarios story narrative") == 2


def test_codex_batch_dry_run_writes_report_schema_and_prompts(tmp_path: Path) -> None:
    bundles = [
        _bundle("joint39_val_0001", split="train", text="A broad risk-off tape."),
        _bundle("joint39_val_0002", split="test", text="A calmer risk-on tape."),
    ]
    report_path = tmp_path / "pipeline_report.json"
    report_path.write_text(json.dumps({"narrative_bundles": bundles}), encoding="utf-8")
    args = Namespace(
        pipeline_report=report_path,
        output_dir=tmp_path / "codex_batch",
        count=2,
        offset=0,
        split="all",
        selection_mode="ordered",
        window_id=[],
        model="gpt-5.5",
        reasoning_effort="xhigh",
        timeout_seconds=5,
        skip_existing=True,
        continue_on_error=True,
        dry_run=True,
        batch_size=2,
    )

    result = run_batch(args)
    saved = json.loads(Path(result["artifact_paths"]["report"]).read_text(encoding="utf-8"))

    assert result["status"] == "dry_run"
    assert result["requested_count"] == 2
    assert result["caption_count"] == 0
    assert result["codex_error_count"] == 2
    assert Path(result["artifact_paths"]["strict_schema"]).exists()
    assert saved["errors"][0]["error_type"] == "DryRun"
    assert (
        tmp_path
        / "codex_batch"
        / "prompts"
        / "prompt_batch_000000_joint39_val_0001_to_joint39_val_0002.txt"
    ).exists()
