import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_14_view_variant_pilot as pilot
from experiments.backfill.block_ar import (
    nl_stride5_self_supervised_training_manifest as manifest,
)


def _target_report(tmp_path: Path, *, target_idx: int = 10) -> Path:
    pairs = []
    for offset, view_name in enumerate(pilot.EXPECTED_VIEW_NAMES):
        pairs.append(
            {
                "view_name": view_name,
                "positive_text": f"positive {offset} for target {target_idx}",
                "negative_window_id": f"joint39_train_{200 + offset:04d}",
                "negative_text": f"hard negative {offset} for target {target_idx}",
                "quality_notes": ["directional near miss"],
            }
        )
    report = {
        "status": "pass",
        "target_window_id": f"joint39_train_{target_idx:04d}",
        "local_prose_generated": False,
        "pairs": pairs,
    }
    report_path = tmp_path / f"target_{target_idx}" / "fourteen_view_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return report_path


def _bank_report(tmp_path: Path, report_path: Path, *, target_idx: int = 10) -> Path:
    payload = {
        "schema_version": "stride5_fourteen_view_bank_report_v1",
        "status": "pass",
        "processed_all_targets": True,
        "local_prose_generated": False,
        "records": [
            {
                "status": "pass",
                "target_window_id": f"joint39_train_{target_idx:04d}",
                "report_path": str(report_path),
            }
        ],
    }
    path = tmp_path / "bank_report.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _support_arrays(tmp_path: Path, *, rows: int = 300) -> Path:
    path = tmp_path / "support_bank_arrays.npz"
    np.savez_compressed(path, memory_targets=np.zeros((rows, 4), dtype=np.float32))
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_manifest_labels_positive_to_target_and_negative_to_negative_window(
    tmp_path: Path,
) -> None:
    target_report = _target_report(tmp_path, target_idx=10)
    bank_report = _bank_report(tmp_path, target_report, target_idx=10)

    report = manifest.build_self_supervised_training_manifest(
        bank_report_path=bank_report,
        output_dir=tmp_path / "out",
        support_arrays_path=_support_arrays(tmp_path),
    )

    assert report["status"] == "pass"
    assert report["summary"]["target_count"] == 1
    assert report["summary"]["pair_count"] == 14
    assert report["summary"]["positive_example_count"] == 14
    assert report["summary"]["hard_negative_example_count"] == 14
    examples = _read_jsonl(Path(report["artifact_paths"]["training_examples_jsonl"]))
    first_positive = examples[0]
    first_negative = examples[1]
    assert first_positive["role"] == "positive"
    assert first_positive["label_window_id"] == "joint39_train_0010"
    assert first_positive["label_window_index"] == 10
    assert first_negative["role"] == "hard_negative"
    assert first_negative["label_window_id"] == "joint39_train_0200"
    assert first_negative["label_window_index"] == 200
    assert first_negative["paired_positive_window_id"] == "joint39_train_0010"


def test_manifest_rejects_duplicate_positive_texts_per_target(tmp_path: Path) -> None:
    target_report = _target_report(tmp_path, target_idx=10)
    payload = json.loads(target_report.read_text(encoding="utf-8"))
    payload["pairs"][1]["positive_text"] = payload["pairs"][0]["positive_text"]
    target_report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    bank_report = _bank_report(tmp_path, target_report, target_idx=10)

    report = manifest.build_self_supervised_training_manifest(
        bank_report_path=bank_report,
        output_dir=tmp_path / "out",
        support_arrays_path=_support_arrays(tmp_path),
    )

    assert report["status"] == "fail"
    assert any(
        error["code"] == "duplicate_positive_texts"
        for error in report["validation"]["errors"]
    )


def test_manifest_checks_memory_target_coverage_for_negative_labels(
    tmp_path: Path,
) -> None:
    target_report = _target_report(tmp_path, target_idx=10)
    bank_report = _bank_report(tmp_path, target_report, target_idx=10)

    report = manifest.build_self_supervised_training_manifest(
        bank_report_path=bank_report,
        output_dir=tmp_path / "out",
        support_arrays_path=_support_arrays(tmp_path, rows=50),
    )

    assert report["status"] == "fail"
    assert any(
        error["code"] == "memory_label_out_of_range"
        and error["window_id"] == "joint39_train_0200"
        for error in report["validation"]["errors"]
    )
