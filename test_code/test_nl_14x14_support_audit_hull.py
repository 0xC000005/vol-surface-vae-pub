"""Integration test for the ADDITIVE framework-v1 §I hull support-honesty gate
wired into the 14x14 support audit (`nl_14x14_support_audit.build_14x14_support_audit`).

Asserts:
  - With a synthetic `grounding_by_window` for the query window, the query_review
    gains a populated `hull_support_honesty_14anchor` block with status "ok" and a
    kappa ladder.
  - Omitting grounding entirely leaves the existing query_review shape unchanged
    (no `hull_support_honesty_14anchor` key) — non-regression / graceful degrade.
  - Supplying grounding but for a different window yields the graceful
    `no_grounding_for_window` status.

The synthetic support bank (`support_bank_arrays.npz`) is written as a SIBLING of the
synthetic support report so `build_anchor_move_pool` runs fully hermetically against
fixtures (no dependency on the real 939a bank).
"""

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


def _example(idx: int, *, window: int, role: str, view: str = "full_professional") -> dict:
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


def _write_synthetic_support_bank(path: Path, *, n_windows: int = 6, seed: int = 7) -> None:
    """Write a synthetic support_bank_arrays.npz that build_anchor_move_pool can consume.

    future_delta has shape (W, 30, 39); the gate reads the terminal step over anchor
    cols 25..38. A nonzero-variance random pool gives a well-posed covariance and a real
    (non-degenerate) feasibility LP.
    """
    rng = np.random.default_rng(seed)
    future_delta = rng.normal(0.0, 1.0, size=(n_windows, 30, 39)).astype(np.float64)
    train_indices = np.arange(n_windows - 1, dtype=np.int64)  # hold one out of train
    np.savez_compressed(
        path,
        future_delta=future_delta,
        train_indices=train_indices,
    )


def _synthetic_grounding() -> dict:
    """Condition-only-grounding shaped object with named-anchor implications.

    Mirrors the real 808c condition_only_grounding schema: SPX up, VIX down, BBB_OAS
    tighter. narrative_emphasis unwraps the `condition_only_grounding` key, so the
    resulting emphasis is non-empty -> status "ok" with a graded ladder.
    """
    return {
        "window_index": 4,
        "condition_only_grounding": {
            "current_market_state_implications": [
                {"market": "SPX", "direction": "up", "magnitude": "medium"},
                {"market": "VIX", "direction": "down", "magnitude": "medium"},
                {"market": "BBB_OAS", "direction": "tighter", "magnitude": "small"},
            ],
        },
    }


def _build_fixture(tmp_path: Path) -> dict:
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
            {"window_index": 0, "window_id": "joint39_train_0000", "manifest_split": "support_train"},
            {"window_index": 1, "window_id": "joint39_train_0001", "manifest_split": "support_train"},
            {"window_index": 2, "window_id": "joint39_train_0002", "manifest_split": "support_train"},
            {"window_index": 3, "window_id": "joint39_train_0003", "manifest_split": "support_train"},
            {"window_index": 4, "window_id": "joint39_train_0004", "manifest_split": "support_decoder_test"},
        ]
    }
    support_report_path = tmp_path / "support_report.json"
    support_report_path.write_text(json.dumps(support_report), encoding="utf-8")

    # The hull gate derives the bank path as <support_report dir>/support_bank_arrays.npz.
    _write_synthetic_support_bank(tmp_path / "support_bank_arrays.npz")

    text_space_arrays = tmp_path / "text_space.npz"
    embeddings = np.asarray(
        [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.7, 0.3, 0.0]],
        dtype=np.float32,
    )
    adapted = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.95, 0.05], [0.85, 0.15]],
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
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.2, 0.8], [0.95, 0.05]],
        dtype=np.float32,
    )
    condition_vectors = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.95, 0.05], [0.88, 0.12]],
        dtype=np.float32,
    )
    np.savez_compressed(
        projected_arrays,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        labels=np.asarray([0, 1, 2, 4, 1], dtype=np.int64),
    )

    return {
        "examples_jsonl": examples_jsonl,
        "support_report_path": support_report_path,
        "text_space_arrays_path": text_space_arrays,
        "projected_arrays_path": projected_arrays,
    }


def test_hull_gate_attaches_ok_ladder_with_grounding(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    output = build_14x14_support_audit(
        examples_jsonl=fixture["examples_jsonl"],
        support_report_path=fixture["support_report_path"],
        text_space_arrays_path=fixture["text_space_arrays_path"],
        projected_arrays_path=fixture["projected_arrays_path"],
        output_dir=tmp_path / "out_with_grounding",
        query_window_indices=[4],
        top_k=3,
        support_pool_size=3,
        temporal_gap=0,
        baseline_bridge_reports={},
        grounding_by_window={4: _synthetic_grounding()},
    )

    # Schema version reflects the additive field; hull_gate_config recorded.
    assert output["schema_version"] == "nl_14x14_support_level_audit_v2_hull_gate"
    assert output["hull_gate_config"]["kappas"] == [0.5, 1.0, 2.0]
    assert output["hull_gate_config"]["scope"] == "14-anchor"

    reviews = output["query_reviews"]
    assert len(reviews) == 1
    block = reviews[0]["hull_support_honesty_14anchor"]
    assert block["status"] == "ok"

    ladder = block["ladder"]
    assert len(ladder) == 3
    assert [rung["kappa"] for rung in ladder] == [0.5, 1.0, 2.0]
    for rung in ladder:
        # Graded signals present and well-typed (the binary `feasible` is a flag only).
        assert set(rung) == {
            "kappa",
            "feasible",
            "l1_distance_sigma",
            "pool_mahalanobis",
            "support_label",
        }
        assert rung["l1_distance_sigma"] is not None
        assert rung["pool_mahalanobis"] is not None
        assert rung["support_label"] in {
            "historically_grounded",
            "outside_historical_analogue_support",
            "indeterminate_lp_failure",
        }
    assert "any_infeasible" in block
    assert "leaves_hull_at_kappa" in block
    # Emphasis was extracted from the named-anchor implications (SPX/VIX/BBB_OAS).
    assert {k.upper() for k in block["emphasis"]} >= {"SPX", "VIX", "BBB_OAS"}
    assert "14 named anchors only" in block["scope_note"]


def test_hull_gate_absent_without_grounding(tmp_path: Path) -> None:
    """Non-regression: omitting grounding leaves the query_review shape untouched."""
    fixture = _build_fixture(tmp_path)
    output = build_14x14_support_audit(
        examples_jsonl=fixture["examples_jsonl"],
        support_report_path=fixture["support_report_path"],
        text_space_arrays_path=fixture["text_space_arrays_path"],
        projected_arrays_path=fixture["projected_arrays_path"],
        output_dir=tmp_path / "out_no_grounding",
        query_window_indices=[4],
        top_k=3,
        support_pool_size=3,
        temporal_gap=0,
        baseline_bridge_reports={},
        # grounding_by_window omitted -> None
    )
    assert output["status"] == "pass"
    reviews = output["query_reviews"]
    assert len(reviews) == 1
    # No hull block at all when grounding is not provided (existing behavior preserved).
    assert "hull_support_honesty_14anchor" not in reviews[0]


def test_hull_gate_graceful_when_window_missing(tmp_path: Path) -> None:
    """Grounding provided but not for this window -> graceful no_grounding_for_window."""
    fixture = _build_fixture(tmp_path)
    grounding = _synthetic_grounding()
    grounding["window_index"] = 999  # a different window
    output = build_14x14_support_audit(
        examples_jsonl=fixture["examples_jsonl"],
        support_report_path=fixture["support_report_path"],
        text_space_arrays_path=fixture["text_space_arrays_path"],
        projected_arrays_path=fixture["projected_arrays_path"],
        output_dir=tmp_path / "out_missing_window",
        query_window_indices=[4],
        top_k=3,
        support_pool_size=3,
        temporal_gap=0,
        baseline_bridge_reports={},
        grounding_by_window={999: grounding},
    )
    block = output["query_reviews"][0]["hull_support_honesty_14anchor"]
    assert block == {"status": "no_grounding_for_window"}
