"""Tests for the P3 locality-first pool-then-reweight chassis.

New memory_prior_mode 'start_pool_text_tilt':
  1. POOL: top-M candidates by start_only_score (no text influence).
  2. TILT: re-weight within the pool only (neutral / memory_cosine / external).
  3. POOL-QUALITY GATE: analogue_scarce flag always attached.
  4. CONFLICT DETECTOR: narrative_start_conflict block always attached.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (
    build_mixture_memory_prior,
)


def _spec_names() -> list[str]:
    return [f"iv:{idx}" for idx in range(25)] + ["factor:spx", "factor:vix"]


def _history(n: int = 6) -> np.ndarray:
    """Synthetic windows with strictly increasing start distance to window 0.

    Every candidate prefix is risk-on (SPX terminal delta +1, VIX terminal
    delta -1) so the conflict detector can be driven purely by the grounding.
    """

    history = np.zeros((n, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    for idx in range(n):
        history[idx, -1, 0] = 0.2 + float(idx)
    history[:, -1, 25] = 1.0
    history[:, -1, 26] = -1.0
    return history


def _memory_targets(n: int = 6) -> np.ndarray:
    # Window n-1 has the highest cosine to the query memory [1, 0] so that
    # memory similarity disagrees with start similarity.
    targets = np.zeros((n, 2), dtype=np.float32)
    for idx in range(n):
        targets[idx, 0] = 0.1 + 0.15 * idx
        targets[idx, 1] = 1.0 - 0.15 * idx
    return targets


def _grounding_risk_on() -> dict:
    return {
        "market_implications": [
            {"market": "SPX", "direction": "up", "confidence": "high"},
            {"market": "VIX", "direction": "down", "confidence": "high"},
        ]
    }


def _grounding_risk_off() -> dict:
    return {
        "market_implications": [
            {"market": "SPX", "direction": "down", "confidence": "high"},
            {"market": "VIX", "direction": "up", "confidence": "high"},
        ]
    }


def _build(mode: str, **overrides) -> dict:
    n = int(overrides.pop("n_windows", 6))
    kwargs = dict(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=_memory_targets(n),
        history_level=_history(n),
        train_indices=np.arange(n, dtype=np.int64),
        query_window_index=0,
        query_start_state=_history(n)[0, -1, :],
        grounding=_grounding_risk_on(),
        spec_names=_spec_names(),
        mode=mode,
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=1.0,
        start_distance_penalty=0.5,
        implication_alignment_weight=0.25,
        diverse_max_pairwise_cosine=0.99,
    )
    kwargs.update(overrides)
    return build_mixture_memory_prior(**kwargs)


# (a) neutral tilt == soft_topk_start_only equivalence ------------------------


def test_neutral_tilt_reproduces_soft_topk_start_only_exactly() -> None:
    baseline = _build("soft_topk_start_only")
    pooled = _build(
        "start_pool_text_tilt",
        pool_size=6,
        tilt_mode="neutral",
        tilt_weight=1.0,
    )

    assert pooled["mode"] == "start_pool_text_tilt"
    assert pooled["window_indices"] == baseline["window_indices"]
    np.testing.assert_allclose(
        pooled["weights"], baseline["weights"], rtol=0.0, atol=0.0
    )


def test_neutral_tilt_equivalence_holds_with_default_pool_size() -> None:
    baseline = _build("soft_topk_start_only")
    pooled = _build("start_pool_text_tilt")  # pool_size defaults to 50 >= top_k

    assert pooled["window_indices"] == baseline["window_indices"]
    np.testing.assert_allclose(
        pooled["weights"], baseline["weights"], rtol=0.0, atol=0.0
    )


# (b) pool membership is locality-only ----------------------------------------


def test_extreme_external_tilt_cannot_pull_window_into_pool() -> None:
    result = _build(
        "start_pool_text_tilt",
        pool_size=3,
        top_k=3,
        tilt_mode="external",
        external_tilt_scores={5: 1.0e9, 4: 1.0e9},
    )

    assert 5 not in result["window_indices"]
    assert 4 not in result["window_indices"]
    assert set(result["window_indices"]) <= {0, 1, 2}


# (c) external tilt reorders within the pool ----------------------------------


def test_external_tilt_reorders_selection_within_pool() -> None:
    neutral = _build("start_pool_text_tilt", pool_size=6, top_k=2)
    tilted = _build(
        "start_pool_text_tilt",
        pool_size=6,
        top_k=2,
        tilt_mode="external",
        tilt_weight=1.0,
        external_tilt_scores={2: 100.0},
    )

    assert neutral["window_indices"][0] == 0
    assert tilted["window_indices"][0] == 2
    assert tilted["weights"][0] == max(tilted["weights"])


def test_memory_cosine_tilt_uses_bridge_signal() -> None:
    # Window 5 has by far the best memory cosine; a huge tilt_weight must be
    # able to promote it within a full pool, but not change pool membership.
    result = _build(
        "start_pool_text_tilt",
        pool_size=6,
        top_k=2,
        tilt_mode="memory_cosine",
        tilt_weight=1000.0,
    )

    assert result["window_indices"][0] == 5


# (d) pool-quality gate --------------------------------------------------------


def test_pool_quality_flags_far_start_query_as_analogue_scarce() -> None:
    far_start = _history()[0, -1, :].copy()
    far_start[0] = 50.0

    result = _build(
        "start_pool_text_tilt",
        query_start_state=far_start,
        pool_size=3,
        pool_reference_quantiles={"q90_min_distance": 3.0},
    )

    quality = result["pool_quality"]
    assert quality["analogue_scarce"] is True
    assert quality["pool_min_start_distance_z"] > 3.0
    assert quality["reference_source"] == "provided_reference_quantiles"


def test_pool_quality_does_not_flag_near_start_query() -> None:
    result = _build(
        "start_pool_text_tilt",
        pool_size=3,
        pool_reference_quantiles={"q90_min_distance": 3.0},
    )

    quality = result["pool_quality"]
    assert quality["analogue_scarce"] is False
    assert quality["pool_min_start_distance_z"] == pytest.approx(0.0, abs=1e-6)


def test_pool_quality_threshold_kwarg_overrides_reference() -> None:
    result = _build(
        "start_pool_text_tilt",
        pool_size=3,
        pool_reference_quantiles={"q90_min_distance": 1.0e9},
        analogue_scarce_threshold_z=-1.0,
    )

    quality = result["pool_quality"]
    assert quality["analogue_scarce_threshold_z"] == -1.0
    assert quality["analogue_scarce"] is True


def test_pool_quality_computes_reference_from_candidate_table_when_missing() -> None:
    result = _build("start_pool_text_tilt", pool_size=3)

    quality = result["pool_quality"]
    assert quality["reference_source"] == "computed_from_candidate_table_q90"
    assert np.isfinite(quality["reference_q90_min_distance"])
    assert isinstance(quality["analogue_scarce"], bool)
    assert quality["pool_median_start_distance_z"] >= 0.0
    assert quality["pool_density"] >= 0.0


# (e) conflict detector ---------------------------------------------------------


def test_conflict_detector_fires_when_grounding_contradicts_pool() -> None:
    result = _build(
        "start_pool_text_tilt",
        grounding=_grounding_risk_off(),
        pool_size=4,
    )

    conflict = result["narrative_start_conflict"]
    assert conflict["checked"] == 2
    assert conflict["mismatch_rate"] == 1.0
    assert conflict["conflict"] is True


def test_conflict_detector_quiet_when_grounding_agrees_with_pool() -> None:
    result = _build(
        "start_pool_text_tilt",
        grounding=_grounding_risk_on(),
        pool_size=4,
    )

    conflict = result["narrative_start_conflict"]
    assert conflict["checked"] == 2
    assert conflict["mismatch_rate"] == 0.0
    assert conflict["conflict"] is False


def test_conflict_detector_handles_missing_grounding() -> None:
    for grounding in (None, {}):
        result = _build(
            "start_pool_text_tilt",
            grounding=grounding,
            pool_size=4,
        )
        conflict = result["narrative_start_conflict"]
        assert conflict["checked"] == 0
        assert conflict["conflict"] is False


# (f) min_index_gap machinery ----------------------------------------------------


def test_min_index_gap_respected_within_pool() -> None:
    result = _build(
        "start_pool_text_tilt",
        pool_size=6,
        top_k=3,
        diverse_min_index_gap=2,
    )

    indices = result["window_indices"]
    assert indices == [0, 2, 4]
    for left, right in zip(indices, indices[1:], strict=False):
        assert abs(right - left) >= 2


# (g) schema superset-compatibility -----------------------------------------------


def test_schema_is_superset_of_soft_topk_start_only() -> None:
    baseline = _build("soft_topk_start_only")
    pooled = _build("start_pool_text_tilt", pool_size=6)

    assert set(baseline.keys()) <= set(pooled.keys())
    assert "pool_quality" in pooled
    assert "narrative_start_conflict" in pooled
    for key in (
        "pool_min_start_distance_z",
        "pool_median_start_distance_z",
        "pool_density",
        "analogue_scarce",
    ):
        assert key in pooled["pool_quality"]
    for key in ("mismatch_rate", "conflict", "checked"):
        assert key in pooled["narrative_start_conflict"]
    assert pooled["analogue_count"] == len(pooled["window_indices"])
    assert abs(sum(pooled["weights"]) - 1.0) < 1e-6
    assert pooled["memory"].shape == baseline["memory"].shape


def test_existing_modes_do_not_grow_new_blocks() -> None:
    baseline = _build("soft_topk_start_only")

    assert "pool_quality" not in baseline
    assert "narrative_start_conflict" not in baseline


# input validation ------------------------------------------------------------------


def test_unknown_tilt_mode_raises() -> None:
    with pytest.raises(ValueError, match="tilt_mode"):
        _build("start_pool_text_tilt", tilt_mode="bogus")


def test_external_tilt_requires_scores() -> None:
    with pytest.raises(ValueError, match="external_tilt_scores"):
        _build("start_pool_text_tilt", tilt_mode="external")
