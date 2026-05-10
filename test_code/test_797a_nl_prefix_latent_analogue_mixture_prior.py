import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (
    build_mixture_memory_prior,
    candidate_support_table,
    direction_check_for_mixture,
    run_analogue_mixture_prior,
    start_distances_to_query_start,
    weighted_prefix_terminal_rows,
)
from experiments.backfill.block_ar.nl_prefix_latent_market_alignment import (
    is_terminal_direction_checkable,
    market_implication_alignment,
)


def _spec_names() -> list[str]:
    return [f"iv:{idx}" for idx in range(25)] + ["factor:spx", "factor:vix"]


def _history() -> np.ndarray:
    history = np.zeros((3, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    # Candidate 0 matches risk-on: SPX up, VIX down.
    history[0, -1, 25] = 2.0
    history[0, -1, 26] = -1.0
    # Candidate 1 is opposite.
    history[1, -1, 25] = -2.0
    history[1, -1, 26] = 1.0
    # Candidate 2 is partly aligned.
    history[2, -1, 25] = 1.0
    history[2, -1, 26] = 1.0
    return history


def _grounding() -> dict:
    return {
        "market_implications": [
            {"market": "SPX", "direction": "up", "confidence": "high"},
            {"market": "VIX", "direction": "down", "confidence": "high"},
        ]
    }


def test_weighted_prefix_terminal_rows_blends_selected_analogues() -> None:
    rows = weighted_prefix_terminal_rows(
        history_level=_history(),
        window_indices=np.asarray([0, 1]),
        weights=np.asarray([0.75, 0.25], dtype=np.float32),
        spec_names=_spec_names(),
    )

    by_market = {row["Market"]: row["Mean Terminal Delta"] for row in rows}

    assert by_market["SPX"] > 0.0
    assert by_market["VIX"] < 0.0


def test_candidate_support_table_combines_memory_and_implication_alignment() -> None:
    rows = candidate_support_table(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=1.0,
    )

    by_idx = {row["window_index"]: row for row in rows}

    assert by_idx[0]["recent_prefix_alignment_score"] == 1.0
    assert by_idx[1]["recent_prefix_alignment_score"] == -1.0
    assert by_idx[1]["narrative_start_score"] > by_idx[0]["narrative_start_score"]
    assert by_idx[1]["recent_prefix_alignment"]["mismatch_count"] == 2
    assert by_idx[0]["combined_score"] > by_idx[1]["combined_score"]


def test_candidate_support_table_can_condition_on_supplied_start_state() -> None:
    query_start = _history()[2, -1, :]

    distances = start_distances_to_query_start(
        start_state=_history()[:, -1, :],
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        query_window_index=0,
        query_start_state=query_start,
    )
    rows = candidate_support_table(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        query_start_state=query_start,
        grounding=_grounding(),
        spec_names=_spec_names(),
        start_distance_threshold_z=0.0,
        start_distance_penalty=10.0,
        implication_alignment_weight=0.0,
    )

    by_idx = {row["window_index"]: row for row in rows}

    assert distances[2] == 0.0
    assert by_idx[2]["start_distance_z"] == 0.0
    assert by_idx[0]["start_distance_cost"] > 0.0
    assert by_idx[2]["combined_score"] > by_idx[0]["combined_score"]


def test_build_mixture_memory_prior_returns_weighted_memory_and_support() -> None:
    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="soft_topk_combined",
        top_k=2,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=1.0,
        diverse_max_pairwise_cosine=0.99,
    )

    assert result["mode"] == "soft_topk_combined"
    assert result["memory"].shape == (2,)
    assert result["analogue_count"] == 2
    assert abs(sum(result["weights"]) - 1.0) < 1e-6
    assert result["support_alignment"]["checked_count"] == 2
    assert result["direction_check"]["status"] == "pass"
    assert result["query_start_source"] == "query_window_index"


def test_narrative_start_mode_keeps_grounding_as_direction_check_only() -> None:
    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="soft_topk_narrative_start",
        top_k=1,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=100.0,
        diverse_max_pairwise_cosine=0.99,
    )

    assert result["window_indices"] == [1]
    assert result["candidate_details"][0]["recent_prefix_mismatches"] == 2
    assert result["direction_check"]["status"] == "reject"
    assert (
        result["direction_check"]["reason"]
        == "final_mixed_prefix_direction_mismatch"
    )


def test_narrative_start_checked_mode_uses_grounding_as_hard_gate() -> None:
    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="soft_topk_narrative_start_checked",
        top_k=1,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=100.0,
        diverse_max_pairwise_cosine=0.99,
    )

    assert result["window_indices"] == [0]
    assert result["candidate_details"][0]["recent_prefix_mismatches"] == 0
    assert result["direction_check"]["status"] == "pass"


def test_direction_check_warns_on_weak_support_before_final_mismatch() -> None:
    check = direction_check_for_mixture(
        candidate_details=[
            {
                "window_index": 0,
                "recent_prefix_checked": 2,
                "recent_prefix_match_count": 1,
                "recent_prefix_mismatches": 1,
                "recent_prefix_alignment_status": "warning",
            }
        ],
        weights=np.asarray([1.0], dtype=np.float32),
        final_mixture_alignment={
            "checked_count": 2,
            "match_count": 2,
            "mismatch_count": 0,
            "status": "pass",
        },
        min_support_match_rate=0.75,
    )

    assert check["status"] == "warning"
    assert check["reason"] == "selected_support_direction_weak"


def test_weighted_rows_are_compatible_with_alignment_helper() -> None:
    rows = weighted_prefix_terminal_rows(
        history_level=_history(),
        window_indices=np.asarray([0, 2]),
        weights=np.asarray([0.8, 0.2], dtype=np.float32),
        spec_names=_spec_names(),
    )

    alignment = market_implication_alignment(
        grounding=_grounding(),
        scenario_rows=rows,
    )

    assert alignment["status"] == "pass"
    assert alignment["mismatch_count"] == 0


def test_market_alignment_skips_static_current_state_level_language() -> None:
    assert not is_terminal_direction_checkable(
        {
            "market": "VIX",
            "direction": "up",
            "horizon": "current_state",
            "evidence": ["volatility remains elevated"],
        }
    )
    alignment = market_implication_alignment(
        grounding={
            "market_implications": [
                {
                    "market": "VIX",
                    "direction": "up",
                    "horizon": "current_state",
                    "evidence": ["volatility remains elevated"],
                },
                {
                    "market": "USDJPY",
                    "direction": "up",
                    "horizon": "current_state",
                    "evidence": ["USDJPY is moving higher"],
                },
            ]
        },
        scenario_rows=[
            {"Market": "VIX", "Mean Terminal Delta": -1.0},
            {"Market": "USDJPY", "Mean Terminal Delta": 1.0},
        ],
    )

    assert alignment["checked_count"] == 1
    assert alignment["skipped_count"] == 1
    assert alignment["status"] == "pass"


def test_run_analogue_mixture_prior_writes_summary(tmp_path) -> None:
    report_path = tmp_path / "case" / "prefix_latent_story_smoke_report.json"
    arrays_path = tmp_path / "case" / "prefix_latent_story_smoke_arrays.npz"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "cached_query": {
                    "window_index": 0,
                    "grounding": _grounding(),
                }
            }
        ),
        encoding="utf-8",
    )
    np.savez(arrays_path, text_memory=np.asarray([[1.0, 0.0]], dtype=np.float32))
    casebook_path = tmp_path / "casebook.json"
    casebook_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "risk_on",
                        "story": "risk-on",
                        "selected_start_status": "warning",
                        "market_alignment": {
                            "checked_count": 2,
                            "mismatch_count": 1,
                        },
                        "artifact_paths": {"prefix_report": str(report_path)},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    oracle_path = tmp_path / "oracle.npz"
    np.savez(
        oracle_path,
        history_level=_history(),
        true_memory=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"state_specs": [{"name": name} for name in _spec_names()]},
        checkpoint_path,
    )

    summary = run_analogue_mixture_prior(
        SimpleNamespace(
            casebook_summary=str(casebook_path),
            oracle_arrays=str(oracle_path),
            checkpoint=str(checkpoint_path),
            output_dir=str(tmp_path / "out"),
            top_k=2,
            temperature=0.2,
            start_distance_threshold_z=100.0,
            start_distance_penalty=0.0,
            implication_alignment_weight=1.0,
            diverse_max_pairwise_cosine=0.99,
        )
    )

    assert summary["status"] == "ok"
    assert summary["case_count"] == 1
    assert "soft_topk_combined" in summary["variant_totals"]
    assert (tmp_path / "out" / "analogue_mixture_prior_summary.json").exists()
