from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_14x14_support_audit import (
    build_14x14_support_audit,
)


def _example(
    idx: int,
    *,
    window: int,
    role: str,
    view: str = "full_professional",
) -> dict:
    return {
        "example_id": f"joint39_train_{window:04d}__{view}__{role}_{idx}",
        "target_window_id": f"joint39_train_{window:04d}",
        "target_window_index": int(window),
        "label_window_id": f"joint39_train_{window:04d}",
        "label_window_index": int(window),
        "role": role,
        "view_name": view,
        "text": f"{role} text for window {window}",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_support_audit_emits_top3_90_bridge_rows(tmp_path: Path) -> None:
    examples = [
        _example(0, window=0, role="positive"),
        _example(1, window=1, role="positive"),
        _example(2, window=2, role="positive"),
        _example(3, window=4, role="positive"),
        {**_example(4, window=1, role="hard_negative"), "label_window_index": 1},
    ]
    examples_jsonl = tmp_path / "examples.jsonl"
    _write_jsonl(examples_jsonl, examples)

    support_report = {
        "window_metadata": [
            {
                "window_index": 0,
                "window_id": "joint39_train_0000",
                "manifest_split": "support_train",
            },
            {
                "window_index": 1,
                "window_id": "joint39_train_0001",
                "manifest_split": "support_train",
            },
            {
                "window_index": 2,
                "window_id": "joint39_train_0002",
                "manifest_split": "support_train",
            },
            {
                "window_index": 3,
                "window_id": "joint39_train_0003",
                "manifest_split": "support_train",
            },
            {
                "window_index": 4,
                "window_id": "joint39_train_0004",
                "manifest_split": "support_decoder_test",
            },
        ]
    }
    support_report_path = tmp_path / "support_report.json"
    support_report_path.write_text(json.dumps(support_report), encoding="utf-8")

    text_space_arrays = tmp_path / "text_space.npz"
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.7, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    adapted = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.95, 0.05],
            [0.85, 0.15],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        text_space_arrays,
        embeddings=embeddings,
        adapted_embeddings=adapted,
        labels=np.asarray([0, 1, 2, 4, 1], dtype=np.int64),
    )

    projected_arrays = tmp_path / "projected.npz"
    memory_targets = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.2, 0.8],
            [0.95, 0.05],
        ],
        dtype=np.float32,
    )
    condition_vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.95, 0.05],
            [0.88, 0.12],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        projected_arrays,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        labels=np.asarray([0, 1, 2, 4, 1], dtype=np.int64),
    )

    output = build_14x14_support_audit(
        examples_jsonl=examples_jsonl,
        support_report_path=support_report_path,
        text_space_arrays_path=text_space_arrays,
        projected_arrays_path=projected_arrays,
        output_dir=tmp_path / "out",
        query_window_indices=[4],
        top_k=3,
        support_pool_size=3,
        temporal_gap=0,
        baseline_bridge_reports={},
    )

    assert output["status"] == "pass"
    assert output["query_window_indices"] == [4]
    assert set(output["bridge_reports"]) == {
        "raw_openai_14x14_top3_90",
        "text_space_contrastive_14x14_top3_90",
        "projected_memory_14x14_top3_90",
    }

    combined_path = Path(output["artifact_paths"]["matched_bridge_report"])
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    rows = combined["evaluation"]["heldout_examples"]
    assert len(rows) == 3
    assert {row["window_index"] for row in rows} == {4}
    for row in rows:
        assert row["role"] == "anchor"
        assert 1 <= len(row["top_train_pool"]) <= 3
        weights = [item["weight"] for item in row["top_train_pool"]]
        assert np.isclose(sum(weights), 1.0)
        assert all(item["posterior_role"] == "top3_90_selected" for item in row["top_train_pool"])

    audit_md = Path(output["artifact_paths"]["markdown"])
    assert "raw_openai_14x14_top3_90" in audit_md.read_text(encoding="utf-8")


def test_support_audit_filters_existing_baseline_to_matched_windows(tmp_path: Path) -> None:
    examples = [
        _example(0, window=0, role="positive"),
        _example(1, window=4, role="positive"),
    ]
    examples_jsonl = tmp_path / "examples.jsonl"
    _write_jsonl(examples_jsonl, examples)
    support_report = {
        "window_metadata": [
            {"window_index": 0, "window_id": "joint39_train_0000", "manifest_split": "support_train"},
            {"window_index": 4, "window_id": "joint39_train_0004", "manifest_split": "support_decoder_test"},
        ]
    }
    support_report_path = tmp_path / "support_report.json"
    support_report_path.write_text(json.dumps(support_report), encoding="utf-8")
    text_space_arrays = tmp_path / "text_space.npz"
    np.savez_compressed(
        text_space_arrays,
        embeddings=np.eye(2, dtype=np.float32),
        adapted_embeddings=np.eye(2, dtype=np.float32),
        labels=np.asarray([0, 4], dtype=np.int64),
    )
    projected_arrays = tmp_path / "projected.npz"
    np.savez_compressed(
        projected_arrays,
        condition_vectors=np.eye(2, dtype=np.float32),
        memory_targets=np.eye(5, 2, dtype=np.float32),
        labels=np.asarray([0, 4], dtype=np.int64),
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "heldout_examples": [
                        {
                            "query_id": "baseline_4",
                            "window_index": 4,
                            "window_id": "joint39_train_0004",
                            "role": "anchor",
                            "kind": "baseline_kind",
                            "top_train_pool": [
                                {"rank": 1, "window_index": 0, "window_id": "joint39_train_0000", "cosine": 0.9, "weight": 0.8},
                            ],
                        },
                        {
                            "query_id": "baseline_5",
                            "window_index": 5,
                            "role": "anchor",
                            "kind": "baseline_kind",
                            "top_train_pool": [],
                        },
                    ]
                },
                "split": {"train_indices": [0], "test_indices": [4]},
            }
        ),
        encoding="utf-8",
    )

    output = build_14x14_support_audit(
        examples_jsonl=examples_jsonl,
        support_report_path=support_report_path,
        text_space_arrays_path=text_space_arrays,
        projected_arrays_path=projected_arrays,
        output_dir=tmp_path / "out",
        query_window_indices=[4],
        top_k=1,
        support_pool_size=1,
        temporal_gap=0,
        baseline_bridge_reports={"baseline_existing": baseline_path},
    )
    combined = json.loads(
        Path(output["artifact_paths"]["matched_bridge_report"]).read_text(encoding="utf-8")
    )
    kinds = [row["kind"] for row in combined["evaluation"]["heldout_examples"]]
    assert "baseline_existing" in kinds
    assert kinds.count("baseline_existing") == 1
